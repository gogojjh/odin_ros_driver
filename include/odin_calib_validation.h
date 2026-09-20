#ifndef ODIN_CALIB_VALIDATION_H
#define ODIN_CALIB_VALIDATION_H

#include <yaml-cpp/yaml.h>
#include <cmath>
#include <string>

namespace odin_ros_driver {

// Match the fields consumed by MultiSensorPublisher and rawCloudRender. File
// existence and a successful SDK return code do not guarantee usable data.
inline bool ValidateOdinCalibFile(const std::string& path, std::string& reason) {
    try {
        const YAML::Node root = YAML::LoadFile(path);
        if (!root.IsMap()) {
            reason = "calibration root must be a mapping";
            return false;
        }
        const YAML::Node cam = root["cam_0"];
        if (!cam.IsMap()) {
            reason = "missing or invalid cam_0";
            return false;
        }
        for (const char* key : {"image_width", "image_height"}) {
            if (!cam[key].IsScalar() || cam[key].as<int>() <= 0) {
                reason = std::string("invalid positive integer cam_0/") + key;
                return false;
            }
        }
        for (const char* key : {"A11", "A12", "A22", "u0", "v0",
                                "k2", "k3", "k4", "k5", "k6", "k7", "p1", "p2"}) {
            const YAML::Node value = cam[key];
            if (!value.IsScalar() || !std::isfinite(value.as<double>()) ||
                !std::isfinite(value.as<float>())) {
                reason = std::string("invalid finite number cam_0/") + key;
                return false;
            }
        }
        if (cam["A11"].as<double>() <= 0 || cam["A22"].as<double>() <= 0) {
            reason = "cam_0 focal lengths must be positive";
            return false;
        }
        const YAML::Node transform = root["Tcl_0"];
        if (!transform.IsSequence() || transform.size() != 16) {
            reason = "Tcl_0 must contain 16 numbers";
            return false;
        }
        for (std::size_t i = 0; i < transform.size(); ++i) {
            if (!transform[i].IsScalar() || !std::isfinite(transform[i].as<double>()) ||
                !std::isfinite(transform[i].as<float>())) {
                reason = "Tcl_0 must contain finite numbers";
                return false;
            }
        }
        reason.clear();
        return true;
    } catch (const std::exception& error) {
        reason = error.what();
        return false;
    }
}

}  // namespace odin_ros_driver
#endif
