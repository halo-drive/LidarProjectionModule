#!/usr/bin/env bash
# Create / refresh per-module unit test workspaces in .unit_dev/

set -e
ROOT=$(cd "$(dirname "$0")/.."; pwd)          # repo root (LidarProjectionLane)
DEV=${ROOT}/.unit_dev

MODULES=(camera_stitching lidar_processing lane_detection fusion)

echo "Creating unit dev workspaces under ${DEV}"
mkdir -p "${DEV}"

for mod in "${MODULES[@]}"; do
  ws="${DEV}/${mod}_ws"
  src="${ws}/src"
  pkg_link="${src}/${mod}"

  echo "  · ${mod}"
  mkdir -p "${src}"
  # refresh symlink (remove if exists but wrong)
  if [ -L "${pkg_link}" ] || [ -e "${pkg_link}" ]; then
    rm -rf "${pkg_link}"
  fi
  ln -s "${ROOT}/${mod}" "${pkg_link}"
done

echo "All unit workspaces ready."
echo "Run   source .unit_dev/<module>_ws/devel/setup.bash   after building."
