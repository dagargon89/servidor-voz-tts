#!/bin/bash
# Configura la voz (TTS) de Open WebUI en su base de datos: detiene el servicio, respalda la base, actualiza y arranca.
set -e
DB=/home/dgarcia/open-webui/data/webui.db
systemctl stop open-webui.service
sudo -u dgarcia cp -a "$DB" "$DB.respaldo-$(date +%Y%m%d-%H%M%S)"
sudo -u dgarcia bash -c '. /home/dgarcia/tts/tts.env; /home/dgarcia/open-webui/.venv/bin/python -I - "$TTS_API_KEY" <<"EOF"
import json, sqlite3, sys, time
c = sqlite3.connect("/home/dgarcia/open-webui/data/webui.db")
valores = {
    "audio.tts.engine": "openai",
    "audio.tts.openai.api_base_url": "http://127.0.0.1:8880/v1",
    "audio.tts.openai.api_key": sys.argv[1],
    "audio.tts.model": "kokoro",
    "audio.tts.voice": "ef_dora",
}
for k, v in valores.items():
    n = c.execute("update config set value=?, updated_at=? where key=?", (json.dumps(v), int(time.time()), k)).rowcount
    print(k, "actualizado" if n else "NO ENCONTRADO")
c.commit()
EOF'
systemctl start open-webui.service
for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null && break; sleep 3; done
systemctl is-active open-webui.service
echo "=== LISTO"
