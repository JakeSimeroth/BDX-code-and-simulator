#!/usr/bin/env bash
# BDX Pi 5 — desktop "appliance" setup:
#   - boot to desktop with autologin
#   - autostart Chromium fullscreen on the Jellyfin UI
#   - install Firefox for YouTube/Google (reachable via Alt+Tab)
#   - disable screen blanking
# NOTE: uses --start-fullscreen (NOT --kiosk) on purpose, so Alt+Tab still works.
set -euo pipefail

echo "==> Installing browsers (Chromium for Jellyfin, Firefox for YouTube)..."
sudo apt update
sudo apt install -y chromium-browser firefox-esr

echo "==> Enabling desktop autologin..."
sudo raspi-config nonint do_boot_behaviour B4 || true

echo "==> Disabling screen blanking..."
sudo raspi-config nonint do_blanking 1 || true   # 1 = disable blanking

echo "==> Autostart Chromium -> Jellyfin on login..."
mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/jellyfin.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Jellyfin
Exec=chromium-browser --start-fullscreen --app=http://localhost:8096
X-GNOME-Autostart-enabled=true
EOF

cat <<'EOF'

==> Desktop appliance configured.
    - On reboot, the Pi boots into the Jellyfin UI (Chromium, fullscreen).
    - Press Alt+Tab to switch to Firefox for YouTube/Google.

    FINISH THE ADBLOCKER (one manual step):
      Open Firefox once and install uBlock Origin:
        https://addons.mozilla.org/firefox/addon/ublock-origin/
      Pi-hole already blocks ads network-wide; uBlock in Firefox adds in-page
      pop-up blocking and YouTube video-ad blocking that DNS can't do.

    Apply now:  sudo reboot
EOF
