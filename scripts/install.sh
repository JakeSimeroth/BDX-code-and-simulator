#!/usr/bin/env bash
# BDX Pi 5 Home Media Server — installer
# Installs Docker (if needed), creates folders, and brings up the stack.
# Run it, edit the generated .env, then run it again.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> BDX media server installer"

# 1. Docker
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "    Added $USER to the docker group (log out/in later to use docker without sudo)."
fi

# 2. .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "==> Created .env from template."
  echo "    EDIT IT NOW (TZ, PIHOLE_PASSWORD, DATA_ROOT), then run this script again:"
  echo "      nano .env && ./scripts/install.sh"
  exit 0
fi

# shellcheck disable=SC1091
set -a; . ./.env; set +a

# 3. Folders
echo "==> Creating folders under '${CONFIG_ROOT}' and '${DATA_ROOT}'..."
mkdir -p "${CONFIG_ROOT}"/jellyfin "${CONFIG_ROOT}"/qbittorrent \
         "${CONFIG_ROOT}"/sonarr "${CONFIG_ROOT}"/pihole/etc-pihole
mkdir -p "${DATA_ROOT}"/media/movies "${DATA_ROOT}"/media/tv "${DATA_ROOT}"/torrents

# 4. Warn if port 53 is taken (Pi-hole needs it; often systemd-resolved squats on it)
if sudo ss -tulpn 2>/dev/null | grep -q ':53 '; then
  echo "WARNING: port 53 is already in use (often systemd-resolved)."
  echo "         Pi-hole needs it. See README section 'Pi-hole and port 53'."
fi

# 5. Start
echo "==> Starting the stack..."
sudo docker compose up -d

IP=$(hostname -I | awk '{print $1}')
cat <<EOF

==> Done. Services:
      Jellyfin       http://${IP}:8096
      Pi-hole admin  http://${IP}:8081/admin     (password from .env)
      qBittorrent    http://${IP}:8080           (temp password: sudo docker logs qbittorrent)
      Sonarr         http://${IP}:8989

    Next: run  ./scripts/setup-kiosk.sh  to boot the Pi straight into Jellyfin.
    Reminder: only add LEGAL sources to qBittorrent/Sonarr — see README "Legal".
EOF
