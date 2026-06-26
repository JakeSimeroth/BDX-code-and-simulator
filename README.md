# BDX Pi 5 Home Media Server

An always-on (24/7) home media server for a Raspberry Pi 5 — packaged as a Docker
stack plus a desktop "appliance" setup that boots straight into Jellyfin.

## What's in the box

| Service | Role | Default URL |
|---|---|---|
| **Jellyfin** | Media server for your movies | `http://<pi-ip>:8096` |
| **Pi-hole** | Network-wide ad / tracker / pop-up blocker | `http://<pi-ip>:8081/admin` |
| **qBittorrent** | BitTorrent client *(general tool — see Legal)* | `http://<pi-ip>:8080` |
| **Sonarr** | Library / PVR manager *(general tool — see Legal)* | `http://<pi-ip>:8989` |
| Desktop | Boots into the Jellyfin UI; **Alt+Tab → Firefox** for YouTube/Google | on the TV |

## Legal — read this

This stack ships **qBittorrent and Sonarr as general-purpose, legal software with no
indexers, trackers, or content sources configured.** It does **not** include, download,
or point to any pirated content, and the indexer-manager layer (Prowlarr/Jackett) that
turns these into an automated downloader is intentionally **not** included.

BitTorrent and Sonarr have legitimate uses — Linux distributions, the Internet Archive,
public-domain films, media you own, and paid/legal Usenet. **Downloading copyrighted
movies or shows you don't have a licence for is copyright infringement**, and that choice
and its consequences are yours. Fill Jellyfin from legal sources: rip discs you own,
public-domain archives (archive.org), or your own files.

## Hardware

- Raspberry Pi 5 (8GB recommended) + **active cooler** (required — the Pi 5 throttles without one)
- Official 27W USB-C power supply, micro-HDMI → HDMI cable
- **Storage: a 1–2TB NVMe SSD on an M.2 HAT, used as the boot drive** (see [Storage](#storage))

## Quick start

```bash
git clone <this-repo> bdx-server && cd bdx-server
./scripts/install.sh        # installs Docker, then creates .env
nano .env                   # set TZ, PIHOLE_PASSWORD, and DATA_ROOT
./scripts/install.sh        # run again: creates folders + starts the stack
```

Then, on the Pi connected to your TV:

```bash
./scripts/setup-kiosk.sh    # boot-into-Jellyfin + Firefox for YouTube
sudo reboot
```

First-run notes:
- **qBittorrent** prints a temporary admin password in its log: `sudo docker logs qbittorrent`.
- **Jellyfin** setup wizard: create your admin user, then add a *Movies* library pointing at `/data/media/movies`.
- **Sonarr/qBittorrent** both see `DATA_ROOT` as `/data`, so downloads and the library share one filesystem (atomic moves / hardlinks).

## Storage

For a 1–2TB target, use **NVMe via an M.2 HAT** rather than USB, and **boot from it**:

- M.2 HAT+ (~$20) + 1–2TB NVMe **M-key** 2242 SSD (~$70–130).
- Move the OS off the microSD: flash Raspberry Pi OS to the NVMe with Raspberry Pi
  Imager, then set the boot order:
  ```bash
  sudo rpi-eeprom-config --edit      # set BOOT_ORDER=0xf416  (NVMe first, SD fallback)
  ```
- Point `DATA_ROOT` in `.env` at the NVMe mount (e.g. `/mnt/storage`). The microSD
  becomes a recovery/backup card.

**Why not microSD:** Jellyfin + Sonarr + qBittorrent + Pi-hole write constantly
(databases, torrent metadata, logs). microSD cards wear out under that load. NVMe is
faster and far more durable.

## Adblocker — two layers (the genuinely "best" setup)

1. **Pi-hole** blocks ads/trackers/pop-ups **network-wide** over DNS — for the Pi and
   every device you point at it. After setup, set the Pi (or your router's DHCP) to use
   the Pi's IP as its DNS server.
2. **uBlock Origin in Firefox** handles what DNS cannot: in-page cosmetic ads, many
   pop-ups, and **YouTube video ads** (these come from the same domains as the video, so
   Pi-hole structurally can't block them).
   - **Use Firefox, not Chromium:** Chrome's Manifest V3 crippled uBlock Origin on
     Chromium (only the weaker "Lite" remains); Firefox still runs the full version.
     Install from <https://addons.mozilla.org/firefox/addon/ublock-origin/>.

## Pi-hole and port 53

Pi-hole needs port 53 (DNS). If `sudo ss -tulpn | grep ':53'` shows it is already taken
(commonly `systemd-resolved`), free it before starting Pi-hole — disable the resolver's
DNS stub listener and set a fallback upstream DNS, then `sudo docker compose up -d pihole`.
Follow the Pi-hole docs for your OS version.

## Updating & 24/7 operation

```bash
sudo docker compose pull && sudo docker compose up -d   # update all containers
```

Every service uses `restart: unless-stopped`, so the stack comes back automatically after
a reboot or power blip. Shut the Pi down cleanly (short-press the onboard power button) —
don't yank wall power, which risks filesystem corruption.

## Remote access (optional)

Reach Jellyfin from anywhere without exposing any ports, using **Tailscale**:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Install the Tailscale app on your phone/laptop (same account), then open
`http://<pi-tailscale-ip>:8096`. Remote playback is limited by your home **upload** speed;
keep movies as 1080p H.264 so the Pi direct-plays instead of transcoding.

## Ports reference

| Port | Service |
|---|---|
| 8096 | Jellyfin web UI |
| 8081 | Pi-hole admin (`/admin`) |
| 53   | Pi-hole DNS |
| 8080 | qBittorrent web UI |
| 8989 | Sonarr web UI |
| 6881 | qBittorrent torrent traffic |
