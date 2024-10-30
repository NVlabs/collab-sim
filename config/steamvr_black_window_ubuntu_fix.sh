#!/bin/bash
# Requires:
# sudo apt-get install jq

# Edit a JSON file using jq
edit_json() {
    local file=$1
    local key_path=$2
    local key=$3
    local value=$4

    if [ -f "$file" ]; then
        # Check if the key exists before editing
        if jq -e ".$key_path.$key" "$file" >/dev/null 2>&1; then
            echo "Modifying '$key' in $file"
            # Safely edit in place without overwriting the whole file
            sudo jq ".$key_path.$key = $value" "$file" > /tmp/temp.json && sudo mv /tmp/temp.json "$file"
        else
            echo "'$key' not found in $file, skipping."
        fi
    else
        echo "$file not found, skipping."
    fi
}

# 1. Remove empty entry "" in the DesktopUI section of steamvr.vrsettings
config_file="$HOME/.steam/steam/config/steamvr.vrsettings"
if [ -f "$config_file" ]; then
    echo "Checking $config_file for empty entry in DesktopUI..."
    sudo jq 'del(.DesktopUI."")' "$config_file" > /tmp/temp.json && sudo mv /tmp/temp.json "$config_file"
else
    echo "$config_file not found, skipping."
fi

# 2. Set preload to false in pairing in the lighthouse webhelperoverlays.json
lighthouse_file="$HOME/.local/share/Steam/steamapps/common/SteamVR/drivers/lighthouse/resources/webhelperoverlays.json"
edit_json "$lighthouse_file" "pairing" "preload" "false"

# 3. Set preload to false in settings_desktop in the SteamVR webhelperoverlays.json
steamvr_file="$HOME/.local/share/Steam/steamapps/common/SteamVR/resources/webhelperoverlays.json"
edit_json "$steamvr_file" "settings_desktop" "preload" "false"

# 4. Set preload to false in Vrlink_pairing in the vrlink webhelperoverlays.json
vrlink_file="$HOME/.local/share/Steam/steamapps/common/SteamVR/drivers/vrlink/resources/webhelperoverlays.json"
edit_json "$vrlink_file" "Vrlink_pairing" "preload" "false"

echo "Completed"
