import pandas as pd
submission = pd.read_csv('/content/sample_submission.csv')

import json
with open('/content/result.jsonl', 'r') as file:
  answer = []
  for line in file:
    info = json.loads(line)
    answer.append(info['text'].strip())

submission['answer'] = pd.DataFrame(answer)
submission.to_csv('llava_submission.csv', index=False)