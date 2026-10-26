#!/bin/bash
# Instala el servidor de voz como servicio (puerto 8880, red local) y reinicia Open WebUI para que lo use.
set -e
cat > /etc/systemd/system/tts.service <<'UNIT'
[Unit]
Description=Servidor de voz TTS (Kokoro, Piper, Qwen3-TTS) compatible con OpenAI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=dgarcia
Group=dgarcia
WorkingDirectory=/home/dgarcia/tts
EnvironmentFile=/home/dgarcia/tts/tts.env
ExecStart=/home/dgarcia/tts/.venv/bin/uvicorn servidor:app --host 0.0.0.0 --port 8880
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now tts.service
if systemctl is-active --quiet ufw; then
  ufw allow from 192.168.3.0/24 to any port 8880 proto tcp comment 'TTS LAN'
fi
systemctl restart open-webui.service
for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:8880/health >/dev/null && break; sleep 2; done
systemctl is-active tts.service open-webui.service
echo "=== LISTO"
