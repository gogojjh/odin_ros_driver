#!/usr/bin/env python3
"""把 Odin1 的相机内参当成 ROS 的 CameraInfo 发出去。

**为什么需要这个节点**：Odin 驱动本身不发 CameraInfo（2026-08-28 实测：它发布的
13 个话题里一条都没有），内参只躺在 `~/.ros/odin_ros_driver/calib.yaml` 里。
以前下游是各自把这几个数抄一份到自己的代码/launch 里，靠注释写着"必须逐位一致"
来维持——抄错了不会报错，只会让地图慢慢长歪。这个节点让内参有一个唯一的源头。

**发的是"去畸变之后那张图"的内参**，所以畸变系数 D 必须全零。
`calib.yaml` 里 `cam_0` 的 `cam_model` 写的是 `FishPoly`（鱼眼多项式），乍看会
以为去畸变之后内参要变。实际不会：驱动的 `buildUndistortMap()`
（`odin_ros_driver/include/host_sdk_sample.h`）是拿**同一组** fx/fy/cx/cy 反算每个
输出像素该到原图哪里取值，只抹掉畸变和 skew（A12）。也就是说
`/odin1/image/undistorted` 就是一台内参恰好等于 A11/A22/u0/v0 的理想针孔相机拍的。
D 要是填上 k2..k7，下游会再去畸变一次，等于把图弄歪。

同一份内参也适用于深度图：驱动 README 写明 `depth_img_competetion` 与
`image_undistort` 是 one-to-one 像素对应。

按 10 Hz 周期发布，同时开 latch。

两个一起开是有讲究的：内参本身是不变的，latch 保证晚起来的订阅者一连上就立刻
拿到、不用等；而周期发布让 `rostopic hz /odin1/camera_info` 能测出频率来 ——
纯 latch 的话 hz 永远是空的（它测的是周期性到达间隔），排查时很容易误判成
"这个话题没在发"。10 Hz 的代价可以忽略（一条消息几百字节）。
频率用 `~rate` 调，设 0 就退回"只 latch 发一次"。
"""
import math
import os
import sys
import time
from typing import Optional, Tuple

import rospy
import yaml
from sensor_msgs.msg import CameraInfo

def default_calib_file() -> str:
    """与 C++ odin_calib_path.h 使用同一套路径优先级。"""
    if "ODIN_CALIB_DIR" in os.environ:
        return os.environ["ODIN_CALIB_DIR"] + "/calib.yaml"
    if "ROS_HOME" in os.environ:
        return os.environ["ROS_HOME"] + "/odin_ros_driver/calib.yaml"
    if "HOME" in os.environ:
        return os.environ["HOME"] + "/.ros/odin_ros_driver/calib.yaml"
    return "/tmp/odin_ros_driver/calib.yaml"


DEFAULT_CALIB = default_calib_file()


def _finite_number(cam: dict, key: str) -> float:
    value = cam[key]
    # bool 是 int 的子类，但不是合法的内参。保留旧版对数值字符串的兼容。
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("cam_0.%s 必须是数值" % key)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("cam_0.%s 必须是有限数值" % key)
    return result


def _load_cam0(path: str) -> Tuple[int, int, float, float, float, float]:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("cam_0"), dict):
        raise ValueError("缺少有效的 cam_0 标定节")
    cam = cfg["cam_0"]
    w = _finite_number(cam, "image_width")
    h = _finite_number(cam, "image_height")
    if any(v <= 0 or not v.is_integer() or v > 0xffffffff for v in (w, h)):
        raise ValueError("图像宽高必须是有效的正整数")
    fx, fy = (_finite_number(cam, key) for key in ("A11", "A22"))
    if fx <= 0 or fy <= 0:
        raise ValueError("焦距 A11/A22 必须大于零")
    cx, cy = (_finite_number(cam, key) for key in ("u0", "v0"))
    return int(w), int(h), fx, fy, cx, cy


def read_cam0(path: str) -> Optional[Tuple[int, int, float, float, float, float]]:
    """读 calib.yaml 的 cam_0，返回 (宽, 高, fx, fy, cx, cy)。

    A11/A22 是焦距、u0/v0 是主点，这是 Odin 标定文件自己的命名。
    skew（A12，实测约 0.19 像素）去畸变时已经被抹掉了，针孔模型里不带它。
    """
    try:
        return _load_cam0(path)
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, OverflowError) as exc:
        rospy.logerr("[相机内参] %s 尚无有效内参：%s", path, exc)
        return None


def wait_for_cam0(path: str, wait_s: float) -> Optional[Tuple[int, int, float, float, float, float]]:
    """等待有效内容，而非仅等文件出现；写入中/空文件均可恢复。

    驱动尚未启动时 ROS 时钟可能没有推进，因此超时、重试及日志节流
    都使用主机 monotonic 时钟，不能用 rospy.sleep/get_time。
    """
    if not math.isfinite(wait_s) or wait_s < 0:
        raise ValueError("wait_timeout 必须是有限的非负秒数")
    deadline = time.monotonic() + wait_s
    next_warning = 0.0
    while not rospy.is_shutdown():
        try:
            return _load_cam0(path)
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, OverflowError) as exc:
            now = time.monotonic()
            if now >= deadline:
                rospy.logerr("[相机内参] 等了 %.1f 秒，%s 仍无有效内参：%s。"
                             "请检查 Odin 标定下载/写入", wait_s, path, exc)
                return None
            if now >= next_warning:
                rospy.logwarn("[相机内参] 等待有效标定 %s：%s", path, exc)
                next_warning = now + 5.0
            time.sleep(min(0.5, deadline - now))
    return None


def main() -> int:
    rospy.init_node("odin_camera_info_pub")
    path = rospy.get_param("~calib_file", DEFAULT_CALIB)
    # 驱动给图像话题填的 frame_id 是空串（实测），所以这里自己指定一个有意义的。
    frame_id = rospy.get_param("~frame_id", "camera_0")
    topic = rospy.get_param("~topic", "/odin1/camera_info")

    # 标定文件是驱动运行时写出来的，可能比本节点晚。等它，别直接退出。
    try:
        wait_s = float(rospy.get_param("~wait_timeout", 60.0))
        vals = wait_for_cam0(path, wait_s)
    except (TypeError, ValueError, OverflowError) as exc:
        rospy.logerr("[相机内参] 无效的等待配置：%s", exc)
        return 1
    if rospy.is_shutdown():
        return 0
    if vals is None:
        return 1
    w, h, fx, fy, cx, cy = vals

    m = CameraInfo()
    m.header.frame_id = frame_id
    m.width, m.height = w, h
    m.distortion_model = "plumb_bob"
    m.D = [0.0] * 5          # 发的是去畸变后那张图，见文件头说明
    m.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    m.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    rate_hz = float(rospy.get_param("~rate", 10.0))
    pub = rospy.Publisher(topic, CameraInfo, queue_size=1, latch=True)
    m.header.stamp = rospy.Time.now()
    pub.publish(m)
    rospy.loginfo("[相机内参] %s ← %dx%d fx=%.4f fy=%.4f cx=%.4f cy=%.4f"
                  "（读自 %s，%s）",
                  topic, w, h, fx, fy, cx, cy, path,
                  "%.1f Hz + latch" % rate_hz if rate_hz > 0 else "只 latch 一次")

    if rate_hz <= 0:
        rospy.spin()
        return 0

    # 内容一个字节都不变，只换时间戳——下游按 stamp 配对时不至于永远拿到一个
    # 开机时刻的老时间。
    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        m.header.stamp = rospy.Time.now()
        pub.publish(m)
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
