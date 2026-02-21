import os
import json
import random
import sys
import pandas as pd
from multiprocessing import Pool

# Set OpenAI API Key
# import openai
# deployment_name = "<insert deployment name>"
# openai.api_base = f"https://{deployment_name}.openai.azure.com/"
# openai.api_key = "<insert API key>"

from src.utils import get_choices, get_correct_answer
from src.craftmd import craftmd_gpt, craftmd_gpt_baseline, craftmd_gpt_intervention

if __name__ == "__main__":
    
    doctor_model_names = ["gpt-4o-mini-2024-07-18"]
    patient_model_names = ["gpt-4o-mini-2024-07-18"]
    intervening_model = "gpt-4o-mini-2024-07-18"
    dataset = pd.read_csv("./data/usmle_and_derm_dataset.csv", index_col=0)

    dataset = dataset[dataset["category"].isin(["Neurology", "Infectious Diseases"])]
    # print(dataset.head(10))
    print(len(dataset))
    # Set number of threads for parallelization
    num_cpu = 10

    run_baseline = False



    cases = [(dataset.loc[idx,"case_id"], 
                dataset.loc[idx,"case_vignette"], 
                dataset.loc[idx,"category"],
              get_choices(dataset,idx), 
              get_correct_answer(dataset,dataset.loc[idx, "case_id"])) for idx in dataset.index]

    for doctor_model_name in doctor_model_names:
        for patient_model_name in patient_model_names:
            run_id = random.randint(0, 9999)


            if (not run_baseline):
                baseline_folder = "./results/4463_gpt-4o-mini-2024-07-18_gpt-4o-mini-2024-07-18"
                baseline_path = os.path.join(baseline_folder, "transcript.json")
                intervention_id = random.randint(0, 9999)
                intervened_folder = os.path.join(baseline_folder, f"intervened-by-{intervening_model}-{intervention_id}")
                intervened_path = os.path.join(intervened_folder, f"intervened-transcripts.json")
                os.makedirs(os.path.dirname(intervened_path), exist_ok=True)
                agent_conv_history_path = os.path.join(intervened_folder, f"agent_conversation_histories.json")
                os.makedirs(os.path.dirname(agent_conv_history_path), exist_ok=True)

                with open(intervened_path, "w") as f:
                    pass
                with open(baseline_path, 'r') as f:
                    baseline_trajectories = json.load(f)
                # baseline_trajectories = json.loads(baseline_path)
                # print(baseline_trajectories)

                # craftmd_gpt_intervention(case, baseline_trajectory , path_dir, doctor_model_name, patient_model_name)
                baseline_trajectories = [temp for temp in baseline_trajectories if temp != {}]


                with Pool(num_cpu) as p:
                    p.starmap(craftmd_gpt_intervention, [(next((t for t in cases if t[0]== x["case_id"]), None) , x, intervened_folder, doctor_model_name, patient_model_name, intervening_model) for x in baseline_trajectories])
                
                print(f"intervention completed, saved to {intervened_path}")

            else:

                print("running baseline")
                path_dir = f"./results/{run_id}_{doctor_model_name}_{patient_model_name}"
                case_min = 0
                case_max = 2

                with Pool(num_cpu) as p:
                    results = p.starmap(craftmd_gpt_baseline, [(x, path_dir, doctor_model_name, patient_model_name, True) for x in cases[case_min:case_max]])

                print(f"baseline results stored in {path_dir}")
                


            # case_min = 0
            # case_max = 351


            # # craftmd_gpt(cases[0], path_dir, doctor_model_name, patient_model_name, True)

            # with Pool(num_cpu) as p:
            #     results = p.starmap(craftmd_gpt, [(x, path_dir, doctor_model_name, patient_model_name, True) for x in cases[case_min:case_max]])
            
            # # pre_true_count  = sum(pre  for pre, post in results)
            # # post_true_count = sum(post for pre, post in results)

            # # improved_count = post_true_count - pre_true_count
            # # print("improved count:", improved_count)
            # # print("pre intervention accuracy:", pre_true_count, "-->",  pre_true_count/ (case_max - case_min))
            # # print("post intervention accuracy:", post_true_count, "-->",  post_true_count/ (case_max - case_min))
        