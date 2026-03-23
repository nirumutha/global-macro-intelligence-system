# ============================================================
# GMIS 2.0 — MODULE 24 — ECONOMIC CALENDAR ENGINE
# Fetches upcoming high-impact economic events for 14 days
# Sources: ForexFactory JSON (free) + curated India schedule
# Alerts on HIGH impact events within next 24 hours
# ============================================================

import requests
import sqlite3
import pandas as pd
import asyncio
import telegram
import os
import sys
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID', '0'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

# ── Events we care about (keywords to match in FF titles) ────
TRACKED_KEYWORDS = {
    # US events
    'CPI':              ('US CPI',           'HIGH',  '🇺🇸'),
    'Non-Farm':         ('NFP Payrolls',      'HIGH',  '🇺🇸'),
    'Nonfarm':          ('NFP Payrolls',      'HIGH',  '🇺🇸'),
    'FOMC':             ('FOMC Meeting',      'HIGH',  '🇺🇸'),
    'Federal Funds':    ('FOMC Rate Decision','HIGH',  '🇺🇸'),
    'GDP':              ('US GDP',            'HIGH',  '🇺🇸'),
    'PCE':              ('US PCE Inflation',  'HIGH',  '🇺🇸'),
    'Unemployment Claims': ('Jobless Claims', 'MEDIUM','🇺🇸'),
    'Retail Sales':     ('US Retail Sales',   'MEDIUM','🇺🇸'),
    'ISM':              ('ISM PMI',           'MEDIUM','🇺🇸'),
    # India events
    'RBI':              ('RBI MPC Meeting',   'HIGH',  '🇮🇳'),
    'India CPI':        ('India CPI',         'HIGH',  '🇮🇳'),
    'India GDP':        ('India GDP',         'HIGH',  '🇮🇳'),
    'India PMI':        ('India PMI',         'MEDIUM','🇮🇳'),
    # Global
    'OPEC':             ('OPEC Meeting',      'HIGH',  '🛢️'),
    'ECB':              ('ECB Rate Decision', 'HIGH',  '🇪🇺'),
    'BOE':              ('BoE Rate Decision', 'HIGH',  '🇬🇧'),
}

# ── Curated India / OPEC schedule (not in ForexFactory) ──────
# These are known dates for 2025/2026 — update as needed
CURATED_EVENTS = [
    # RBI MPC 2025
    {'event': 'RBI MPC Meeting', 'date': '2025-04-09', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    {'event': 'RBI MPC Meeting', 'date': '2025-06-06', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    {'event': 'RBI MPC Meeting', 'date': '2025-08-06', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    {'event': 'RBI MPC Meeting', 'date': '2025-10-07', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    {'event': 'RBI MPC Meeting', 'date': '2025-12-05', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    # RBI MPC 2026
    {'event': 'RBI MPC Meeting', 'date': '2026-02-07', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    {'event': 'RBI MPC Meeting', 'date': '2026-04-08', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    {'event': 'RBI MPC Meeting', 'date': '2026-06-05', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'RBI'},
    # OPEC meetings 2025/2026
    {'event': 'OPEC Meeting',    'date': '2025-06-01', 'country': 'OPEC',
     'impact': 'HIGH', 'flag': '🛢️',  'source': 'OPEC'},
    {'event': 'OPEC Meeting',    'date': '2025-11-01', 'country': 'OPEC',
     'impact': 'HIGH', 'flag': '🛢️',  'source': 'OPEC'},
    {'event': 'OPEC Meeting',    'date': '2026-06-01', 'country': 'OPEC',
     'impact': 'HIGH', 'flag': '🛢️',  'source': 'OPEC'},
    # India CPI (typically released ~12th of month)
    {'event': 'India CPI',       'date': '2025-04-14', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-05-14', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-06-12', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-07-14', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-08-13', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-09-12', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-10-13', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-11-12', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2025-12-12', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2026-01-14', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2026-02-12', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2026-03-12', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
    {'event': 'India CPI',       'date': '2026-04-14', 'country': 'IN',
     'impact': 'HIGH', 'flag': '🇮🇳', 'source': 'MOSPI'},
]

# ═════════════════════════════════════════════════════════════
# SECTION 1 — FETCH FROM FOREXFACTORY
# ═════════════════════════════════════════════════════════════

def fetch_forexfactory_week(week='thisweek'):
    """
    Fetch ForexFactory calendar JSON for this week or next week.
    Free endpoint, no API key needed.
    Returns list of event dicts.
    """
    url = f'https://nfs.faireconomy.media/ff_calendar_{week}.json'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️ ForexFactory {week} fetch failed: {e}")
        return []


def parse_ff_events(raw_events):
    """
    Parse ForexFactory events and filter to tracked keywords.
    Returns list of standardised event dicts.
    """
    events = []
    today  = datetime.now().date()
    cutoff = today + timedelta(days=14)

    for ev in raw_events:
        title   = ev.get('title', '')
        country = ev.get('country', '').upper()
        impact  = ev.get('impact', '').upper()
        date_str = ev.get('date', '')

        # Skip low/holiday impact unless it matches key keywords
        if impact not in ('High', 'HIGH', 'Medium', 'MEDIUM',
                          'high', 'medium'):
            continue

        # Normalise impact
        impact_norm = 'HIGH'   if impact.lower() == 'high' else 'MEDIUM'

        # Parse date
        try:
            event_date = dateparser.parse(date_str).date()
        except Exception:
            continue

        if event_date < today or event_date > cutoff:
            continue

        # Match against tracked keywords
        matched_name = None
        matched_flag = '🌐'
        for keyword, (name, req_impact, flag) in TRACKED_KEYWORDS.items():
            if keyword.lower() in title.lower():
                matched_name = name
                matched_flag = flag
                # Use our importance level, not FF's
                impact_norm  = req_impact
                break

        # Also keep all HIGH impact USD/EUR/GBP events even if
        # not in our keyword list
        if matched_name is None:
            if impact_norm == 'HIGH' and country in ('US', 'USD'):
                matched_name = title[:60]
                matched_flag = '🇺🇸'
            else:
                continue

        events.append({
            'event':   matched_name,
            'date':    event_date.strftime('%Y-%m-%d'),
            'country': country,
            'impact':  impact_norm,
            'flag':    matched_flag,
            'time':    ev.get('time', 'All Day'),
            'forecast': ev.get('forecast', ''),
            'previous': ev.get('previous', ''),
            'source':  'ForexFactory',
        })

    return events


# ═════════════════════════════════════════════════════════════
# SECTION 2 — MERGE WITH CURATED SCHEDULE
# ═════════════════════════════════════════════════════════════

def get_curated_events_in_window():
    """Return curated events falling within the next 14 days."""
    today  = datetime.now().date()
    cutoff = today + timedelta(days=14)
    result = []

    for ev in CURATED_EVENTS:
        try:
            d = datetime.strptime(ev['date'], '%Y-%m-%d').date()
        except Exception:
            continue
        if today <= d <= cutoff:
            result.append({
                'event':    ev['event'],
                'date':     ev['date'],
                'country':  ev['country'],
                'impact':   ev['impact'],
                'flag':     ev['flag'],
                'time':     'TBD',
                'forecast': '',
                'previous': '',
                'source':   ev['source'],
            })

    return result


def merge_events(ff_events, curated_events):
    """
    Merge ForexFactory and curated events.
    Deduplicate by (event name, date).
    """
    seen   = set()
    merged = []

    for ev in ff_events + curated_events:
        key = (ev['event'].lower()[:30], ev['date'])
        if key not in seen:
            seen.add(key)
            merged.append(ev)

    # Sort by date then impact (HIGH first)
    impact_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    merged.sort(key=lambda x: (
        x['date'],
        impact_order.get(x['impact'], 9)
    ))

    return merged


# ═════════════════════════════════════════════════════════════
# SECTION 3 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_calendar(events):
    """Upsert events into ECONOMIC_CALENDAR table."""
    if not events:
        print("  ⚠️ No events to save")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ECONOMIC_CALENDAR (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event       TEXT NOT NULL,
                date        TEXT NOT NULL,
                country     TEXT,
                impact      TEXT,
                flag        TEXT,
                time        TEXT,
                forecast    TEXT,
                previous    TEXT,
                source      TEXT,
                fetched_on  TEXT,
                UNIQUE(event, date)
            )
        ''')
        conn.commit()

        today_str = datetime.now().strftime('%Y-%m-%d')
        inserted = 0
        updated  = 0

        for ev in events:
            # Try insert; on conflict update
            existing = conn.execute(
                'SELECT id FROM ECONOMIC_CALENDAR WHERE event=? AND date=?',
                (ev['event'], ev['date'])
            ).fetchone()

            if existing:
                conn.execute('''
                    UPDATE ECONOMIC_CALENDAR
                    SET country=?, impact=?, flag=?, time=?,
                        forecast=?, previous=?, source=?,
                        fetched_on=?
                    WHERE event=? AND date=?
                ''', (
                    ev['country'], ev['impact'], ev['flag'],
                    ev['time'], ev['forecast'], ev['previous'],
                    ev['source'], today_str,
                    ev['event'], ev['date']
                ))
                updated += 1
            else:
                conn.execute('''
                    INSERT INTO ECONOMIC_CALENDAR
                    (event, date, country, impact, flag, time,
                     forecast, previous, source, fetched_on)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                ''', (
                    ev['event'], ev['date'], ev['country'],
                    ev['impact'], ev['flag'], ev['time'],
                    ev['forecast'], ev['previous'],
                    ev['source'], today_str
                ))
                inserted += 1

        conn.commit()
        print(f"  ✅ Calendar saved: {inserted} new, {updated} updated")

    except Exception as e:
        print(f"  ❌ Save failed: {e}")
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════
# SECTION 4 — PRINT CALENDAR REPORT
# ═════════════════════════════════════════════════════════════

def print_calendar_report(events):
    """Print upcoming events in a clean table."""
    today  = datetime.now().date()
    cutoff = today + timedelta(days=14)

    print("\n" + "="*70)
    print("ECONOMIC CALENDAR — NEXT 14 DAYS")
    print(f"{today.strftime('%d %b %Y')} → "
          f"{cutoff.strftime('%d %b %Y')}")
    print("="*70)

    if not events:
        print("  No tracked events in the next 14 days")
        return

    print(f"\n{'Date':<12} {'Flag':<3} {'Event':<35} "
          f"{'Impact':<8} {'Time':<10} {'Forecast'}")
    print("-"*80)

    prev_date = None
    for ev in events:
        ev_date = datetime.strptime(ev['date'], '%Y-%m-%d').date()
        days_away = (ev_date - today).days

        # Day separator
        if ev['date'] != prev_date:
            day_label = ev_date.strftime('%a %d %b')
            if days_away == 0:
                day_label += '  ← TODAY'
            elif days_away == 1:
                day_label += '  ← TOMORROW'
            print(f"\n  {day_label}")
            prev_date = ev['date']

        impact_marker = '🔴' if ev['impact'] == 'HIGH' else '🟡'
        forecast_str  = ev.get('forecast', '') or ''
        time_str      = ev.get('time', '') or 'TBD'

        print(f"  {'':2} {ev['flag']:<3} {ev['event']:<35} "
              f"{impact_marker} {ev['impact']:<6} "
              f"{time_str:<10} {forecast_str}")

    # Summary counts
    high_count   = sum(1 for e in events if e['impact'] == 'HIGH')
    medium_count = sum(1 for e in events if e['impact'] == 'MEDIUM')
    in_24h       = [e for e in events
                    if (datetime.strptime(e['date'], '%Y-%m-%d').date()
                        - today).days <= 1]

    print("\n" + "="*70)
    print(f"  TOTAL: {len(events)} events | "
          f"🔴 HIGH: {high_count} | 🟡 MEDIUM: {medium_count}")

    if in_24h:
        print(f"\n  ⚡ EVENTS IN NEXT 24 HOURS:")
        for ev in in_24h:
            print(f"     {ev['flag']} {ev['event']} "
                  f"({ev['date']}) — {ev['impact']}")

    print("="*70)


# ═════════════════════════════════════════════════════════════
# SECTION 5 — TELEGRAM ALERT
# ═════════════════════════════════════════════════════════════

async def send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(events_24h, events_all):
    """Build alert message for HIGH impact events in next 24h."""
    today = datetime.now().date()
    date  = datetime.now().strftime('%d %b %Y %H:%M')

    lines = [
        f"📅 <b>GMIS ECONOMIC CALENDAR</b>",
        f"{date}",
        f"{'─' * 30}",
        f"",
    ]

    if events_24h:
        lines.append(f"⚡ <b>HIGH IMPACT — NEXT 24 HOURS</b>")
        for ev in events_24h:
            ev_date = datetime.strptime(ev['date'], '%Y-%m-%d').date()
            day_str = 'TODAY' if ev_date == today else 'TOMORROW'
            forecast = (f" | Forecast: {ev['forecast']}"
                        if ev.get('forecast') else '')
            lines.append(
                f"  🔴 {ev['flag']} <b>{ev['event']}</b> "
                f"({day_str} {ev.get('time','')}{forecast})"
            )
        lines.append("")

    # Upcoming HIGH events (next 14 days, not in 24h window)
    upcoming_high = [
        e for e in events_all
        if e['impact'] == 'HIGH' and e not in events_24h
    ][:5]  # Top 5

    if upcoming_high:
        lines.append(f"📆 <b>UPCOMING KEY EVENTS (14 days)</b>")
        for ev in upcoming_high:
            ev_date = datetime.strptime(ev['date'], '%Y-%m-%d').date()
            days_away = (ev_date - today).days
            lines.append(
                f"  {ev['flag']} {ev['event']} "
                f"— {ev['date']} ({days_away}d)"
            )
        lines.append("")

    lines.append("<i>GMIS Economic Calendar Engine</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_economic_calendar(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 24 — ECONOMIC CALENDAR ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    # 1. Fetch ForexFactory (this week + next week)
    print("\nFetching ForexFactory calendar...")
    raw_this  = fetch_forexfactory_week('thisweek')
    raw_next  = fetch_forexfactory_week('nextweek')
    ff_events = parse_ff_events(raw_this + raw_next)
    print(f"  ForexFactory: {len(ff_events)} tracked events found")

    # 2. Add curated India/OPEC events
    curated = get_curated_events_in_window()
    print(f"  Curated events: {len(curated)} in next 14 days")

    # 3. Merge and deduplicate
    all_events = merge_events(ff_events, curated)
    print(f"  Total unique events: {len(all_events)}")

    # 4. Save to database
    print("\nSaving to database...")
    save_calendar(all_events)

    # 5. Print report
    print_calendar_report(all_events)

    # 6. Telegram alert for HIGH impact events in next 24 hours
    if send_telegram_flag and BOT_TOKEN:
        today = datetime.now().date()
        events_24h = [
            e for e in all_events
            if e['impact'] == 'HIGH'
            and (datetime.strptime(e['date'], '%Y-%m-%d').date()
                 - today).days <= 1
            and (datetime.strptime(e['date'], '%Y-%m-%d').date()
                 >= today)
        ]

        if '--force-send' in sys.argv:
            events_24h = [e for e in all_events
                          if e['impact'] == 'HIGH'][:3]

        if events_24h:
            print(f"\n  {len(events_24h)} HIGH impact event(s) "
                  f"in next 24h — sending Telegram alert...")
            msg = build_telegram_message(events_24h, all_events)
            asyncio.run(send_telegram(msg))
        else:
            print("\n  No HIGH impact events in next 24h "
                  "— no Telegram alert")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return all_events


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_economic_calendar(send_telegram_flag=not no_telegram)
