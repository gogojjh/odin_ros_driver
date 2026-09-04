/*
Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>


class PointCloudToDepthConverter
{
public:

    struct CameraParams
    {
        int image_width;
        int image_height;
        double A11, A12, A22;          
        double u0, v0;                 
        double k2, k3, k4, k5, k6, k7; 
        double scale;                  
        int point_sampling_rate;      
        Eigen::Matrix4d Tcl;       

        // --- 2026-09-04 追加：投影阶段的三道门。都给类内默认值，任何构造路径
        //     忘了设也不会读到未初始化的垃圾。
        // 小于这个距离的点直接不要（米）。原始云里约 1% 的点在 0.10 m 以内、
        // 6.5% 是 (0,0,0) 占位点，之前只是碰巧投影出界才没进图，不是被有意拒绝。
        double min_range = 0.15;
        // 补洞时允许的邻居深度落差（米）。超过这个值说明跨了深度断崖，不补。
        double hole_fill_tol = 0.15;
        // 补一个洞至少要有几个真测到的邻居。
        int hole_fill_min_neighbors = 3;
    };


    struct ProcessResult
    {
        cv::Mat depth_image;
        pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
        bool success;
        std::string error_message;
    };


    explicit PointCloudToDepthConverter(const CameraParams &params);


    // want_colored_cloud=false 时跳过彩色点云生成（逐点采样上色的双重循环，
    // 成本与深度图本体同量级）——彩色云只用于可视化，没人订阅时是纯浪费。
    ProcessResult processCloudAndImage(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                                       const cv::Mat &image,
                                       bool want_colored_cloud = true);

	cv::Mat customResize(const cv::Mat& src, const cv::Size& size);
    const CameraParams &getCameraParams() const { return params_; }


    void updateCameraParams(const CameraParams &params);

private:
    CameraParams params_;

    Eigen::Matrix3d K_;
    Eigen::Matrix3d Kl_;
    Eigen::Matrix4d K_4x4_;
    Eigen::Matrix4d Kcl_;

    cv::Mat map_x_, map_y_;
    cv::Mat inv_map_x_, inv_map_y_;

    int scaled_width_, scaled_height_;


    void initializeInternalParams();

 
    void createDistortionMaps();

    // cloud_lidar 只用来算真实距离（做最小量程门）；cloud_in_cam 是被 Kcl_
    // 变换过的、x/y 已经乘上内参的齐次像素坐标，两者同序等长。
    cv::Mat projectCloudToDepth(const pcl::PointCloud<pcl::PointXYZ> &cloud_lidar,
                                const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam);


    cv::Mat postProcessDepthImage(const cv::Mat &depth_img);


    pcl::PointCloud<pcl::PointXYZRGB> generateColoredCloud(const cv::Mat &depth_img,
                                                           const cv::Mat &color_img);


    std::pair<bool, std::string> validateInputs(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                                                const cv::Mat &image);
};