// Copyright (C) 2024 retinify project team. All rights reserved.
//
// This file is part of retinify.
//
// retinify is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// retinify is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with retinify.
// If not, see <https://www.gnu.org/licenses/>.

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <libudev.h>
#include <linux/videodev2.h>
#include <map>
#include <retinify/device.hpp>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#define UNKNOWN_DEVICE "Unknown"

static int SafeIOControl(int fd, int request, void *arg)
{
    int result;

    while (true) // loop until success or error is not EINTR
    {
        result = ioctl(fd, request, arg);
        if (result != -1 || errno != EINTR)
        {
            break;
        }
    }

    return result;
}

static void print_pixel_format(uint32_t pixel_format)
{
    switch (pixel_format)
    {
    case V4L2_PIX_FMT_YUYV:
        std::cout << "V4L2_PIX_FMT_YUYV";
        break;
    case V4L2_PIX_FMT_MJPEG:
        std::cout << "V4L2_PIX_FMT_MJPEG";
        break;
    case V4L2_PIX_FMT_H264:
        std::cout << "V4L2_PIX_FMT_H264";
        break;
    case V4L2_PIX_FMT_RGB24:
        std::cout << "V4L2_PIX_FMT_RGB24";
        break;
    case V4L2_PIX_FMT_GREY:
        std::cout << "V4L2_PIX_FMT_GREY";
        break;
    default:
        std::cout << "Unknown format: " << std::hex << pixel_format << std::dec;
        break;
    }
    std::cout << std::endl;
}

static void GetSupportedV4L2Format(retinify::DeviceData &device)
{
    int fd = open(device.node_.c_str(), O_RDWR);
    if (fd == -1)
    {
        std::cerr << "Failed to open device node: " << device.node_ << std::endl;
        return;                                 
    }

    v4l2_fmtdesc fmt_desc;
    memset(&fmt_desc, 0, sizeof(fmt_desc));
    fmt_desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    while (SafeIOControl(fd, VIDIOC_ENUM_FMT, &fmt_desc) == 0)
    {
        v4l2_frmsizeenum frm_size;
        memset(&frm_size, 0, sizeof(frm_size));
        frm_size.pixel_format = fmt_desc.pixelformat;

        print_pixel_format(fmt_desc.pixelformat);

        while (SafeIOControl(fd, VIDIOC_ENUM_FRAMESIZES, &frm_size) == 0)
        {
            if (frm_size.type == V4L2_FRMSIZE_TYPE_DISCRETE)
            {
                device.formats_[fmt_desc.pixelformat].push_back(cv::Size(frm_size.discrete.width, frm_size.discrete.height));
            }
            frm_size.index++;
        }
        fmt_desc.index++;
    }
}

std::map<std::string, retinify::DeviceData> retinify::GetConnectedDeviceMap()
{
    udev *udev = udev_new();
    if (!udev)
    {
        std::cerr << "Failed to create udev object" << std::endl;
        return std::map<std::string, retinify::DeviceData>();
    }

    udev_enumerate *enumerate = udev_enumerate_new(udev);
    if (!enumerate)
    {
        std::cerr << "Failed to create udev enumerate object" << std::endl;
        return std::map<std::string, retinify::DeviceData>();
    }

    udev_enumerate_add_match_subsystem(enumerate, "video4linux");
    udev_enumerate_scan_devices(enumerate);
    udev_list_entry *first_entry = udev_enumerate_get_list_entry(enumerate);
    if (!first_entry)
    {
        std::cerr << "Failed to get device list" << std::endl;
        return std::map<std::string, retinify::DeviceData>();
    }

    udev_list_entry *list_entry;
    std::map<std::string, retinify::DeviceData> device_map;

    udev_list_entry_foreach(list_entry, first_entry)
    {
        const char *path = udev_list_entry_get_name(list_entry);
        udev_device *dev = udev_device_new_from_syspath(udev, path);
        if (dev)
        {
            retinify::DeviceData device;

            if (udev_device_get_devnode(dev))
            {
                device.node_ = udev_device_get_devnode(dev);
            }
            else
            {
                device.node_ = UNKNOWN_DEVICE;
            }

            if (udev_device_get_property_value(dev, "ID_V4L_PRODUCT"))
            {
                device.name_ = udev_device_get_property_value(dev, "ID_V4L_PRODUCT");
            }
            else if (udev_device_get_property_value(dev, "ID_MODEL_FROM_DATABASE"))
            {
                device.name_ = udev_device_get_property_value(dev, "ID_MODEL_FROM_DATABASE");
            }
            else
            {
                device.name_ = UNKNOWN_DEVICE;
            }

            if (udev_device_get_property_value(dev, "ID_V4L_CAPABILITIES"))
            {
                device.capabilities_ = udev_device_get_property_value(dev, "ID_V4L_CAPABILITIES");
            }
            else
            {
                device.capabilities_ = UNKNOWN_DEVICE;
            }

            if (udev_device_get_property_value(dev, "ID_SERIAL"))
            {
                device.serialNumber_ = udev_device_get_property_value(dev, "ID_SERIAL");
            }
            else
            {
                device.serialNumber_ = UNKNOWN_DEVICE;
            }

            if (device.capabilities_.find("capture") != std::string::npos)
            {
                if (device.serialNumber_ != UNKNOWN_DEVICE)
                {
                    GetSupportedV4L2Format(device);
                    device_map[device.serialNumber_] = device;
                }
            }

            udev_device_unref(dev);
        }
    }

    udev_enumerate_unref(enumerate);
    udev_unref(udev);

    return device_map;
}

std::optional<retinify::DeviceData> retinify::GetDeviceBySerialNumber(std::string serialNumber)
{
    std::map<std::string, retinify::DeviceData> devicemap = GetConnectedDeviceMap();
    if (devicemap.find(serialNumber) != devicemap.end())
    {
        return devicemap.at(serialNumber);
    }
    else
    {
        std::cerr << "Device with serial number " << serialNumber << " Not found" << std::endl;
        return std::nullopt;
    }
}