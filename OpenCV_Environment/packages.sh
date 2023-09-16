# Author: Pietro Girotto
# ID: 2088245
export DEBIAN_FRONTEND=noninteractive

packages=("build-essential"
  "cmake"
  "git"
  "pkg-config"
  "libgtk-3-dev"
  "libavcodec-dev"
  "libavformat-dev"
  "libswscale-dev"
  "libv4l-dev"
  "libxvidcore-dev"
  "libx264-dev"
  "libjpeg-dev"
  "libpng-dev"
  "libtiff-dev"
  "gfortran"
  "openexr"
  "libatlas-base-dev"
  "python3-dev"
  "python3-numpy"
  "libtbb2"
  "libtbb-dev"
  "libdc1394-22-dev"
  "libopenexr-dev"
  "libgstreamer-plugins-base1.0-dev"
  "libgstreamer1.0-dev"
)

total_packages=${#packages[@]}
count=0

echo "Starting package installation..."
echo "######################################"

for package in "${packages[@]}"; do

  progress=$((++count * 100 / total_packages))

  echo "Progress: $progress%, trying to install $package"

  if dpkg -l | grep -q "^ii.*$package "; then
    echo "$package is already installed."
  else
    if apt-cache search "^$package$" | grep -q "^$package "; then
      apt-get -qq -y install "$package" > /dev/null
      echo "$package has been installed."
    else
      echo "$package is not found."
    fi
  fi
done

echo "######################################"
