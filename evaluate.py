import pandas as pd
import os

path = "/home/dimitri/master_thesis/real_code/outputs"
csv = [
    f for f in os.listdir(path)
    if os.path.isfile(os.path.join(path, f))
]

for c in csv:
    if not "gemma" in c:
        continue
    df = pd.read_csv('outputs/' + c)
    df['successful'] = df['successful'].astype(int)
    df = df[df['language'] == 'English']
    
    language_success_rate = df.groupby('language')['successful'].mean().reset_index()
    language_success_rate.columns = ['language', 'attack_success_rate']
    language_success_rate_sorted = language_success_rate.sort_values(by='attack_success_rate', ascending=False)

    overall_success_rate = df['successful'].mean()

    print("Reading from Dataset: " + c)    
    print("Attack Success Rate per Language:")
    #print(language_success_rate_sorted.to_string(index=False))

    print("Overall Attack Success Rate: {:.2%}".format(overall_success_rate))
    print("\n")
