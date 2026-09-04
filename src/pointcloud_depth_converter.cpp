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

#include "pointcloud_depth_converter.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

PointCloudToDepthConverter::PointCloudToDepthConverter(const CameraParams &params)
    : params_(params)
{
    initializeInternalParams();
    createDistortionMaps();
}

void PointCloudToDepthConverter::initializeInternalParams()
{
    scaled_width_ = static_cast<int>(params_.image_width / params_.scale);
    scaled_height_ = static_cast<int>(params_.image_height / params_.scale);

    K_ = Eigen::Matrix3d::Identity();
    K_(0, 0) = params_.A11;
    K_(0, 1) = params_.A12;
    K_(0, 2) = params_.u0;
    K_(1, 1) = params_.A22;
    K_(1, 2) = params_.v0;

    Kl_ = Eigen::Matrix3d::Identity();
    Kl_(0, 0) = params_.A11 / params_.scale;
    Kl_(0, 1) = 0.0;
    Kl_(0, 2) = params_.u0 / params_.scale;
    Kl_(1, 1) = params_.A22 / params_.scale;
    Kl_(1, 2) = params_.v0 / params_.scale;

    K_4x4_ = Eigen::Matrix4d::Identity();
    K_4x4_.block<3, 3>(0, 0) = Kl_;

    Kcl_ = K_4x4_ * params_.Tcl;
}

void PointCloudToDepthConverter::createDistortionMaps()
{
    map_x_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);
    map_y_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);


    for (int u = 0; u < params_.image_width; ++u)
    {
        for (int v = 0; v < params_.image_height; ++v)
        {
            double y = (v - params_.v0) / params_.A22;
            double x = (u - params_.u0 - params_.A12 * y) / params_.A11;
            
            double r = sqrt(x * x + y * y);
            double theta = atan(r);

            double theta_d = theta + params_.k2 * pow(theta, 2) + params_.k3 * pow(theta, 3) +
                                params_.k4 * pow(theta, 4) + params_.k5 * pow(theta, 5) +
                                params_.k6 * pow(theta, 6) + params_.k7 * pow(theta, 7);

            double x_distorted = x * (r / theta_d);
            double y_distorted = y * (r / theta_d);

            map_x_.at<float>(v, u) = static_cast<float>(x_distorted * params_.A11 + params_.A12 * y_distorted + params_.u0);
            map_y_.at<float>(v, u) = static_cast<float>(y_distorted * params_.A22 + params_.v0);
        }
    }

    inv_map_x_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);
    inv_map_y_ = cv::Mat::zeros(params_.image_height, params_.image_width, CV_32FC1);
    for (int u = 0; u < params_.image_width; ++u)
    {
        for (int v = 0; v < params_.image_height; ++v)
        {
            double y = (v - params_.v0) / params_.A22;
            double x = (u - params_.u0 - params_.A12 * y) / params_.A11;
            
            double r = sqrt(x * x + y * y);
            double theta = atan(r);

            double theta_d = theta + params_.k2 * pow(theta, 2) + params_.k3 * pow(theta, 3) +
                                params_.k4 * pow(theta, 4) + params_.k5 * pow(theta, 5) +
                                params_.k6 * pow(theta, 6) + params_.k7 * pow(theta, 7);

            double x_distorted = x * (theta_d / r);
            double y_distorted = y * (theta_d / r);

            inv_map_x_.at<float>(v, u) = static_cast<float>(x_distorted * params_.A11 + params_.A12 * y_distorted + params_.u0);
            inv_map_y_.at<float>(v, u) = static_cast<float>(y_distorted * params_.A22 + params_.v0);
        }
    }
}

PointCloudToDepthConverter::ProcessResult PointCloudToDepthConverter::processCloudAndImage(
    const pcl::PointCloud<pcl::PointXYZ> &cloud,
    const cv::Mat &image,
    bool want_colored_cloud)
{
    ProcessResult result;
    result.success = false;

    auto validation_result = validateInputs(cloud, image);
    if (!validation_result.first)
    {
        result.error_message = validation_result.second;
        return result;
    }

    try
    {
        pcl::PointCloud<pcl::PointXYZ> cloud_in_cam;
        pcl::transformPointCloud(cloud, cloud_in_cam, Kcl_);

        cv::Mat depth_img = projectCloudToDepth(cloud, cloud_in_cam);

        cv::Mat processed_depth = postProcessDepthImage(depth_img);

        if (want_colored_cloud)
            result.colored_cloud = generateColoredCloud(processed_depth, image);

        result.depth_image = processed_depth;
        result.success = true;
    }
    catch (const std::exception &e)
    {
        result.error_message = std::string("Processing error: ") + e.what();
    }

    return result;
}

// 2026-09-04 重写。原来这里有两个会凭空造出障碍的毛病：
//   1. 同一个像素被多个点命中时是「后写的赢」，跟远近无关 —— 近点盖住远点就是
//      凭空多一个障碍，远点盖住近点就是真障碍消失。17 帧离线样本实测每帧有
//      290~900 个格子因此拿到错的深度，误差 p90 = 3.25 m、最大 16 m。
//   2. 每个点无条件把自己的深度涂到 8 个邻居上（只要邻居还是 0），跨深度断崖
//      也照涂。结果真有 LiDAR 点的格子只占 ~66%，最终深度图却 91.9% 都「有值」。
// 现在改成两趟：先只写真实测量并取最近的（z-buffer），再单独补洞、且只在局部
// 平坦时才补。没测到就留 0，让下游知道那里是「没观测」而不是「测到了空地」。
cv::Mat PointCloudToDepthConverter::projectCloudToDepth(
    const pcl::PointCloud<pcl::PointXYZ> &cloud_lidar,
    const pcl::PointCloud<pcl::PointXYZ> &cloud_in_cam)
{
    cv::Mat depth_img = cv::Mat::zeros(scaled_height_, scaled_width_, CV_32FC1);
    // measured==1 = 这个格子有真实测量。补洞填出来的格子保持 0，第二趟就不会
    // 拿「编出来的深度」当种子继续往外扩。
    cv::Mat measured = cv::Mat::zeros(scaled_height_, scaled_width_, CV_8UC1);

    const size_t n = std::min(cloud_lidar.size(), cloud_in_cam.size());
    const float min_range_sq = static_cast<float>(params_.min_range * params_.min_range);

    // Pass A：只写真实测量，同一格子最近的赢。
    for (size_t i = 0; i < n; ++i)
    {
        const auto &lp = cloud_lidar[i];
        const auto &camera_point = cloud_in_cam[i];

        if (!std::isfinite(lp.x) || !std::isfinite(lp.y) || !std::isfinite(lp.z))
            continue;
        // 最小量程门。顺带把 (0,0,0) 占位点（原始云里占 6.5%）也一并挡掉。
        if (lp.x * lp.x + lp.y * lp.y + lp.z * lp.z < min_range_sq)
            continue;
        if (camera_point.z <= 0)
            continue;

        const int u = static_cast<int>(std::round(camera_point.x / camera_point.z));
        const int v = static_cast<int>(std::round(camera_point.y / camera_point.z));
        if (u < 0 || u >= scaled_width_ || v < 0 || v >= scaled_height_)
            continue;

        const float z = static_cast<float>(camera_point.z);
        if (measured.at<uchar>(v, u) == 0 || z < depth_img.at<float>(v, u))
        {
            depth_img.at<float>(v, u) = z;
            measured.at<uchar>(v, u) = 1;
        }
    }

    // Pass B：补洞。只补真的没测到的格子，且要求周围够多个真实邻居、彼此深度
    // 落差够小（说明是同一个连续表面上的一个空洞，插值才有意义）。
    cv::Mat filled = depth_img.clone();
    for (int v = 0; v < scaled_height_; ++v)
    {
        for (int u = 0; u < scaled_width_; ++u)
        {
            if (measured.at<uchar>(v, u))
                continue;

            int cnt = 0;
            float lo = 0.0f, hi = 0.0f;
            for (int dv = -1; dv <= 1; ++dv)
            {
                for (int du = -1; du <= 1; ++du)
                {
                    const int nu = u + du, nv = v + dv;
                    if (nu < 0 || nu >= scaled_width_ || nv < 0 || nv >= scaled_height_)
                        continue;
                    if (!measured.at<uchar>(nv, nu))
                        continue;
                    const float d = depth_img.at<float>(nv, nu);
                    if (cnt == 0) { lo = d; hi = d; }
                    else { lo = std::min(lo, d); hi = std::max(hi, d); }
                    ++cnt;
                }
            }

            if (cnt < params_.hole_fill_min_neighbors)
                continue;
            if (hi - lo > static_cast<float>(params_.hole_fill_tol))
                continue;

            // 取最远的那个邻居。这条链路的失效代价是不对称的：凭空多一个近障碍
            // 会把路堵死，而少一格远深度只是让那块暂时算「没观测」。
            filled.at<float>(v, u) = hi;
        }
    }

    return filled;
}

cv::Mat PointCloudToDepthConverter::postProcessDepthImage(const cv::Mat &depth_img) {
    if (depth_img.empty()) {
        std::cerr << "ERROR: Input depth image is empty!" << std::endl;
        return cv::Mat();
    }
    
    if (depth_img.data == nullptr) {
        std::cerr << "ERROR: Input depth image has null data pointer!" << std::endl;
        return cv::Mat();
    }
    
    if (depth_img.rows <= 0 || depth_img.cols <= 0) {
        std::cerr << "ERROR: Invalid input dimensions: " 
                  << depth_img.rows << "x" << depth_img.cols << std::endl;
        return cv::Mat();
    }
    
    if (params_.image_width <= 0 || params_.image_height <= 0) {
        std::cerr << "ERROR: Invalid target size: " 
                  << params_.image_width << "x" << params_.image_height << std::endl;
        return cv::Mat();
    }

    cv::Mat safe_input = depth_img.clone();
    if (safe_input.empty()) {
        std::cerr << "ERROR: Failed to create safe copy of input image!" << std::endl;
        return cv::Mat();
    }
    

    cv::Mat depth_img_upsampled;
    try {
        depth_img_upsampled = customResize(safe_input, cv::Size(1600, 1296));
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Custom resize failed: " << e.what() << std::endl;
        return cv::Mat();
    }
    
    if (depth_img_upsampled.empty()) {
        std::cerr << "ERROR: Resized image is empty!" << std::endl;
        return cv::Mat();
    }
    
    if (depth_img_upsampled.rows != 1296 || depth_img_upsampled.cols != 1600) {
        std::cerr << "ERROR: Resized image has wrong dimensions: " 
                  << depth_img_upsampled.cols << "x" << depth_img_upsampled.rows
                  << " (expected " << 1600 << "x" << 1296 << ")" << std::endl;
        return cv::Mat();
    }
    

    cv::Mat grad_x, grad_y, grad_magnitude;
    try {
        cv::Sobel(depth_img_upsampled, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(depth_img_upsampled, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, grad_magnitude);
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: Sobel/magnitude failed: " << e.what() << std::endl;
        return cv::Mat();
    }
    
    if (grad_magnitude.type() != CV_32F) {
        std::cerr << "ERROR: grad_magnitude has wrong type: " 
                  << grad_magnitude.type() << " (expected CV_32F)" << std::endl;
        return cv::Mat();
    }
    

    cv::Mat threshold_mask;
    try {
        cv::threshold(grad_magnitude, threshold_mask, 0.75, 1, cv::THRESH_BINARY);
        threshold_mask.convertTo(threshold_mask, CV_8U);
        

        depth_img_upsampled.setTo(0, threshold_mask);
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: Threshold mask failed: " << e.what() << std::endl;
        return cv::Mat();
    }

    return depth_img_upsampled;
}

cv::Mat PointCloudToDepthConverter::customResize(const cv::Mat& src, const cv::Size& size) {
    if (src.empty()) {
        throw std::runtime_error("Source image is empty");
    }
    
    if (size.width <= 0 || size.height <= 0) {
        throw std::runtime_error("Invalid target size");
    }
    

    cv::Mat dst(size.height, size.width, src.type());
    
    float scale_x = src.cols / static_cast<float>(size.width);
    float scale_y = src.rows / static_cast<float>(size.height);
    
    if (src.channels() != 1 || src.type() != CV_32F) {
        throw std::runtime_error("Unsupported image type - expected single channel float");
    }
    

    for (int y = 0; y < dst.rows; y++) {

        int src_y = static_cast<int>(y * scale_y);
        src_y = std::min(src_y, src.rows - 1);
        
        for (int x = 0; x < dst.cols; x++) {
            int src_x = static_cast<int>(x * scale_x);
            src_x = std::min(src_x, src.cols - 1);
            dst.at<float>(y, x) = src.at<float>(src_y, src_x);
        }
    }
    
    return dst;
}
pcl::PointCloud<pcl::PointXYZRGB> PointCloudToDepthConverter::generateColoredCloud(
    const cv::Mat &depth_img, const cv::Mat &color_img)
{
    cv::Mat depth_undistorted, color_undistorted;
    depth_undistorted = depth_img.clone();
    cv::remap(color_img, color_undistorted, inv_map_x_, inv_map_y_, cv::INTER_LINEAR);

    pcl::PointCloud<pcl::PointXYZRGB> cloud_colored;

    Eigen::Matrix4d Tlc = params_.Tcl.inverse();

    for (int v = 0; v < depth_undistorted.rows; v += params_.point_sampling_rate)
    {
        for (int u = 0; u < depth_undistorted.cols; u += params_.point_sampling_rate)
        {
            float depth = depth_undistorted.at<float>(v, u);
            if (depth > 0.1f && depth < 100.0f) 
            {
                double y_cam = (v - params_.v0) * depth / params_.A22;
                double x_cam = ((u - params_.u0) * depth  - params_.A12 * y_cam)/ params_.A11;
                
                double z_cam = depth;

                Eigen::Vector4d point_cam(x_cam, y_cam, z_cam, 1.0);

                Eigen::Vector4d point_lidar = Tlc * point_cam;

                pcl::PointXYZRGB point;
                point.x = static_cast<float>(point_lidar[0]);
                point.y = static_cast<float>(point_lidar[1]);
                point.z = static_cast<float>(point_lidar[2]);

                if (u < color_undistorted.cols && v < color_undistorted.rows)
                {
                    cv::Vec3b color = color_undistorted.at<cv::Vec3b>(v, u);
                    point.b = color[0]; 
                    point.g = color[1];
                    point.r = color[2];
                }
                else
                {
                    point.r = point.g = point.b = 255;
                }

                cloud_colored.points.push_back(point);
            }
        }
    }

    cloud_colored.width = cloud_colored.points.size();
    cloud_colored.height = 1;
    cloud_colored.is_dense = false;

    return cloud_colored;
}

std::pair<bool, std::string> PointCloudToDepthConverter::validateInputs(
    const pcl::PointCloud<pcl::PointXYZ> &cloud, const cv::Mat &image)
{
    if (cloud.empty())
    {
        return {false, "Empty point cloud"};
    }

    if (image.empty())
    {
        return {false, "Empty image"};
    }

    if (params_.A11 < 1e-6 || params_.A22 < 1e-6)
    {
        return {false, "Invalid camera intrinsics"};
    }

    return {true, ""};
}

void PointCloudToDepthConverter::updateCameraParams(const CameraParams &params)
{
    params_ = params;
    initializeInternalParams();
    createDistortionMaps();
}
