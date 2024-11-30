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
#include <iostream>
#include <libudev.h>
#include <map>
#include <poll.h>
#include <retinify/device.hpp>
#define UNKNOWN_DEVICE "Unknown"

std::map<std::string, retinify::DeviceData> retinify::EnumerateDevices()
{
    struct udev *udev = udev_new();
    if (!udev)
    {
        std::cerr << "Cannot create udev" << std::endl;
        return std::map<std::string, retinify::DeviceData>();
    }

    struct udev_enumerate *enumerate = udev_enumerate_new(udev);
    if (!enumerate)
    {
        std::cerr << "Failed to create udev enumerate object" << std::endl;
        return std::map<std::string, retinify::DeviceData>();
    }

    udev_enumerate_add_match_subsystem(enumerate, "video4linux");
    udev_enumerate_scan_devices(enumerate);
    struct udev_list_entry *devices = udev_enumerate_get_list_entry(enumerate);
    struct udev_list_entry *entry;

    std::map<std::string, retinify::DeviceData> device_map;

    udev_list_entry_foreach(entry, devices)
    {
        const char *path = udev_list_entry_get_name(entry);
        struct udev_device *dev = udev_device_new_from_syspath(udev, path);
        if (dev)
        {
            retinify::DeviceData device;
            device.node_ = udev_device_get_devnode(dev) ? udev_device_get_devnode(dev) : UNKNOWN_DEVICE;
            device.name_ = udev_device_get_property_value(dev, "ID_V4L_PRODUCT")
                               ? udev_device_get_property_value(dev, "ID_V4L_PRODUCT")
                               : UNKNOWN_DEVICE;
            device.capabilities_ = udev_device_get_property_value(dev, "ID_V4L_CAPABILITIES")
                                       ? udev_device_get_property_value(dev, "ID_V4L_CAPABILITIES")
                                       : UNKNOWN_DEVICE;
            device.serialNumber_ = udev_device_get_property_value(dev, "ID_SERIAL")
                                       ? udev_device_get_property_value(dev, "ID_SERIAL")
                                       : UNKNOWN_DEVICE;
            if (device.capabilities_.find("capture") != std::string::npos)
            {
                if (device.serialNumber_ != UNKNOWN_DEVICE)
                {
                    // set device to map
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
    std::map<std::string, retinify::DeviceData> devicemap = EnumerateDevices();
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