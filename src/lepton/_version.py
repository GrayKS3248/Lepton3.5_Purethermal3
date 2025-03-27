import json

version_json = '''
{
 "date": "2025-03-27T17:00:00-0000",
 "version": "0.0.1"
}
'''


def get_versions():
    return json.loads(version_json)
