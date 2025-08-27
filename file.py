import os
import time
import ftplib
import traceback
from datetime import datetime, timezone

# ===================== CONFIG =====================
FTP_HOST = "192.168.8.41"
FTP_PORT = 21
FTP_USER = "meetech"
FTP_PASS = "123"
REMOTE_DIR = "/myfolder/photobooth"      # remote folder containing images
LOCAL_DATASET = "dataset"                # local folder (no subfolders)
CHECK_INTERVAL_SECONDS = 60              # run every minute
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

# Use local timezone for "today"
TODAY_TZ = datetime.now().astimezone().tzinfo
CONNECT_TIMEOUT = 45
# ==================================================


def is_image(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTS


def today_date(tz):
    return datetime.now(tz).date()


def ensure_local_folder():
    os.makedirs(LOCAL_DATASET, exist_ok=True)


def ftp_connect():
    ftp = ftplib.FTP()
    print("[CONNECT] Plain FTP {}:{} (timeout {}) ...".format(FTP_HOST, FTP_PORT, CONNECT_TIMEOUT))
    ftp.connect(FTP_HOST, FTP_PORT, timeout=CONNECT_TIMEOUT)
    print("[CONNECT] Login ...")
    ftp.login(FTP_USER, FTP_PASS)
    print("[CONNECT] Logged in. PASV=True")
    ftp.set_pasv(True)
    return ftp


def _parse_mlst_modify(mod_raw):
    """
    Parse MLSD 'modify' to UTC datetime.
    Accepts 'YYYYMMDDHHMMSS' or 'YYYYMMDDTHHMMSS' or with fractional '.fff'.
    """
    if not mod_raw:
        return None
    ts = mod_raw.replace("T", "")
    if "." in ts:
        ts = ts.split(".")[0]
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def list_with_mlsd(ftp: ftplib.FTP, remote_dir: str):
    ftp.cwd(remote_dir)
    entries = []
    try:
        ftp.sendcmd("OPTS MLST type;modify;size;")
    except Exception:
        pass
    for name, facts in ftp.mlsd():
        if facts.get("type") != "file":
            continue
        mod_dt = _parse_mlst_modify(facts.get("modify"))
        entries.append({"name": name, "modify": mod_dt})
    return entries


def get_mdtm(ftp: ftplib.FTP, filename: str):
    """
    Return UTC datetime from MDTM. Accepts 'YYYYMMDDHHMMSS' or with fractional '.fff'.
    """
    try:
        resp = ftp.sendcmd("MDTM " + filename)  # e.g., '213 20250810063312' or '213 20250810063312.718'
        parts = resp.strip().split()
        if len(parts) == 2 and parts[0] == "213":
            ts = parts[1]
            if "." in ts:
                ts = ts.split(".")[0]
            return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def filter_today(entries, tz_for_today):
    tdate = today_date(tz_for_today)
    print("\n[DEBUG] Checking files for today's date: {}".format(tdate))
    out = []
    for e in entries:
        if e["modify"] is None:
            print("  [SKIP] {} -> No modify date from server".format(e["name"]))
            continue
        local_dt = e["modify"].astimezone(tz_for_today)
        print("  File: {}".format(e["name"]))
        print("    Server UTC time: {}".format(e["modify"]))
        print("    Local time:      {}".format(local_dt))
        print("    Comparing server date {} with today's date {}".format(local_dt.date(), tdate))
        if local_dt.date() == tdate:
            print("    TODAY? True")
            out.append(e)
        else:
            print("    TODAY? False")
    return out


def download_if_new(ftp: ftplib.FTP, name: str):
    local_path = os.path.join(LOCAL_DATASET, name)
    if os.path.exists(local_path):
        return False
    tmp_path = local_path + ".part"
    with open(tmp_path, "wb") as f:
        ftp.retrbinary("RETR " + name, f.write)
    os.replace(tmp_path, local_path)
    print("[DOWNLOADED] {}".format(name))
    return True


def sync_today_once():
    ensure_local_folder()
    new_count = 0

    ftp = ftp_connect()
    try:
        print("[REMOTE] CWD {}".format(REMOTE_DIR))
        ftp.cwd(REMOTE_DIR)

        try:
            entries = list_with_mlsd(ftp, REMOTE_DIR)
            entries = [e for e in entries if is_image(e["name"])]
            entries = filter_today(entries, TODAY_TZ)
        except ftplib.error_perm:
            print("[INFO] MLSD not supported, using NLST + MDTM.")
            names = ftp.nlst()
            entries = []
            for name in names:
                if not is_image(name):
                    continue
                mod_dt = get_mdtm(ftp, name)
                entries.append({"name": name, "modify": mod_dt})
            entries = filter_today(entries, TODAY_TZ)

        print("[FOUND] {} file(s) for today.".format(len(entries)))
        for e in entries:
            try:
                if download_if_new(ftp, e["name"]):
                    new_count += 1
            except Exception as ex:
                print("[ERROR] Failed to download {}: {}".format(e["name"], ex))
                traceback.print_exc()
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    if new_count:
        print("[SYNC] {} new file(s) downloaded to {}".format(new_count, LOCAL_DATASET))
    else:
        print("[SYNC] No new files.")


def main_loop():
    print("Starting FTP sync loop every {} seconds.".format(CHECK_INTERVAL_SECONDS))
    print("Remote: {}:{}/{}".format(FTP_HOST, FTP_PORT, REMOTE_DIR))
    print("Local:  {}".format(os.path.abspath(LOCAL_DATASET)))
    while True:
        try:
            sync_today_once()
        except Exception as ex:
            print("[ERROR] Sync iteration failed: {}".format(ex))
            traceback.print_exc()
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
