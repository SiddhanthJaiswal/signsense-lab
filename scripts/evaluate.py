"""Exercise real inference and create a review sheet. Does not score ASL automatically."""
import csv
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("greeting", "Hello, how are you?", "Friendly greeting: HELLO HOW YOU; appropriate friendly facial grammar."),
    ("request", "I would like a glass of water.", "Water request; meaning is not replaced by a similar sign."),
    ("negation", "I do not want coffee.", "Negation must change the sentence meaning."),
    ("time", "Tomorrow I will go to school.", "Future/time reference and sentence organization."),
    ("direction_a", "I give you the book.", "Agent is me; recipient is you; compare the reverse sentence."),
    ("direction_b", "You give me the book.", "Agent is you; recipient is me; directional contrast must survive."),
    ("name", "My name is Alex.", "Name handling; do not assume valid fingerspelling."),
]

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def main():
    if not request('/api/health')['ready']:
        raise SystemExit('Model assets are not ready. Finish setup first.')
    output = ROOT / 'runtime/evaluation'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, text, focus in CASES:
        print(f'Generating {key}: {text}', flush=True)
        job = request('/api/translations', {"text": text})
        while job['status'] not in {'complete', 'failed'}:
            time.sleep(2)
            job = request('/api/translations/' + job['id'])
        (output / f'{key}.json').write_text(json.dumps(job, indent=2))
        result = job.get('result') or {}
        rows.append({"case": key, "text": text, "status": job['status'], "review_focus": focus,
                     "duration": result.get('duration'), "semantic_fidelity_0_4": '', "handshape_0_4": '',
                     "orientation_location_0_4": '', "movement_0_4": '', "nonmanual_0_4": '',
                     "reviewer": '', "review_notes": ''})
        print(job['message'], flush=True)
    with (output / 'review.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Review sheet: {output / "review.csv"}. Scores remain blank until human review.')

if __name__ == '__main__':
    main()
