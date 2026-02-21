import copy
import pprint
import os
import json
from copy import deepcopy
from openai import OpenAI
import openai
import docent
from pydantic import Field
from docent import Docent
from typing import Any
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message
from docent.samples import get_inspect_fpath
from docent.data_models import BaseAgentRunMetadata
from pydantic_core import to_jsonable_python
from docent.data_models.chat import SystemMessage, UserMessage, AssistantMessage, ToolMessage, ContentReasoning
from docent.data_models.chat import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ContentText,
    ContentReasoning,
    ToolCall,
)
from .utils import *
from .models import *
from .prompts import *

# improved_count = 0

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class CustomAgentRunMetadata(BaseAgentRunMetadata):
    task_id: str = Field(
        description="The ID of the 'benchmark' or 'set of evals' that the transcript belongs to"
    )

    sample_id: str = Field(
        description="The specific task inside of the `task_id` benchmark that the transcript was run on"
    )
    epoch_id: str = Field(
        description="Each `sample_id` should be run multiple times due to stochasticity; `epoch_id` is the integer index of a specific run."
    )

    model: str = Field(description="The model that was used to generate the transcript")

    scoring_metadata: dict[str, Any] | None = Field(
        description="Additional metadata about the scoring process"
    )

    additional_metadata: dict[str, Any] | None = Field(
        description="Additional metadata about the transcript"
    )


def chat_text(model, messages, max_tokens=4096, temperature=1.0):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content, resp.model_dump()

def run_intervention(transcript, N=5, temperature = 1.0, model = "gpt-4o-mini-2024-07-18"):


    def load_extra_data(transcript):
        return transcript["trial_0"]["multiturn_conversation"][0], transcript["case_vignette"]
    
    def load_TAU_Reasoning_inspect_log(log) -> list[AgentRun]:
        agent_runs: list[AgentRun] = []

        scores: dict[str, int | float | bool] = {}
        scores["correct"] = (log["trial_0"]["reward"])

        metadata = CustomAgentRunMetadata(
            task_id="craftmd",
            sample_id=str(log["case_id"]),
            epoch_id=str(log["case_id"]),
            model=model,
            scores=scores,
            additional_metadata=None,
            scoring_metadata=None,
        )

        messages = []


        for idx, message in enumerate(log["trial_0"]["multiturn_conversation"]):
            if message["role"] == "tool":
                messages.append(ToolMessage(id = str(idx),content=message["content"], tool_call_id=message["tool_call_id"], function=message["name"]))

            elif message["role"] == "assistant":
                contentstr = message["content"]
                if contentstr == None:
                    contentstr = ""
                messages.append(AssistantMessage(id = str(idx),content=contentstr))

            else:
                messages.append(UserMessage(id = str(idx),content=message["content"])) 
            
        agent_runs.append(
            AgentRun(
                transcripts={
                    "default": Transcript(
                        messages=messages
                    )
                },
                metadata=metadata,
            )
        )

        return agent_runs



    def execute_search(text, query, model):

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user","content":SEARCH_PROMPT.format(text=text, search_query=query, SINGLE_RUN_CITE_INSTRUCTION=SINGLE_RUN_CITE_INSTRUCTION)}],
            max_completion_tokens=4096,
            temperature = temperature
        )
        return response.choices[0].message.content.strip()
    

    # print("extra data")
    specification, task = load_extra_data(transcript)
    # print("agent run")
    agent_run_docent = load_TAU_Reasoning_inspect_log(transcript)

    # print("agent run:",agent_run_docent)
    conversation_history = []
    transcript_length = len(transcript["trial_0"]["multiturn_conversation"])
    # print("transcript_length", transcript_length)
    conversation_history.append({
        "role": "system",
        "content": questioning_agent_prompt_working_backwards.format(specification=specification, ref_metadata=task, N=N, transcript_length = transcript_length),
    })   
    
    turns = 0

    while turns < 30:
        try:
            turns+=1
            # print("turn:", turns)
            response = openai.chat.completions.create(
                model=model,
                messages=conversation_history,
                max_completion_tokens=4096,
                temperature=temperature
            )

            reply = response.choices[0].message.content.strip()
            match = re.search(r'<query>(.*?)</query>', reply, re.DOTALL)

            conversation_history.append({
                "role": "assistant",
                "content": reply,
            })



            if match:
                query_text = match.group(1).strip()
                conversation_history[-1]["tool_calls"] = [{"function": { "arguments": query_text, "name": "querying_tool"      }, "id": "12345","type": "function"}]
                tool_response = execute_search(agent_run_docent[0].transcripts["default"].to_str(), query_text, model).strip()

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": "12345",
                    "content": tool_response
                })
                continue



            match = re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL)

            if match:
                answer_text = match.group(1).strip()
                try:
                    

                    answer_list = json.loads(answer_text)

                    possible_new_trajectories = []
                

                    return answer_list, conversation_history

                except json.JSONDecodeError:
                    print("Error decoding JSON.")
                break


            else:
                print("model did not call query tool or generate intervention")
                break
        except Exception as e:
            print(f"error: {e}.")
            break

    

    print("no changes with intervention, error must have happened")
    return [False], conversation_history




def craftmd_gpt(case, path_dir, doctor_model_name, patient_model_name, intervention, num_runs = 1):
    case_id, case_desc, specialty, mcq_choices, case_gt_answer = case
    case_desc, question = get_case_without_question(case_desc)
    # try:

    print(f'Thread for case id: {case_id} dispatched.')

    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt(specialty)
    patient_prompt = get_patient_prompt(case_desc)
    stats = {}  
    j = 0

    save_path = f'{path_dir}/transcript.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
    else:
        os.makedirs(path_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({}, f)


    conv_save_path = f'{path_dir}/agent_conversation_histories.json'
    if not os.path.exists(conv_save_path):
        os.makedirs(path_dir, exist_ok=True)
        with open(conv_save_path, "w") as f:
            json.dump({}, f)

    intervened_path = f'{path_dir}/intervened-transcripts.json'
    if not os.path.exists(intervened_path):
        os.makedirs(path_dir, exist_ok=True)
        with open(intervened_path, "w") as f:
            json.dump({}, f)
    stats = {}
    intervened_stats = {}

    stats["case_id"] = case_id
    stats["case_vignette"] = case_desc
    stats["question"] = question
    




    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt(specialty, case_desc, question, mcq_choices)
        convo = [{"role": "system", "content": vignette_mcq_prompt}]
        vignette_mcq_ans, _ = chat_text(doctor_model_name, convo)

        stats[f"trial_{j}"]["vignette_mcq"] = vignette_mcq_ans
        
        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt(specialty, case_desc, question)
        convo = [{"role": "system", "content": vignette_frq_prompt}]
        vignette_frq_ans, _ = chat_text(doctor_model_name, convo)

        # print("here5")
        stats[f"trial_{j}"]["vignette_frq"] = vignette_frq_ans
        
        # Multi-turn conversation experiments
        conversation_history_doctor = [{"role": "system", "content": doctor_prompt},
                                    {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]

        # print("beginning conversation")
        while True:
            # Patient talks
            # response_patient = patient_model(conversation_history_patient) 

            # response_patient = openai.chat.completions.create(
            #     model=patient_model_name,
            #     messages=conversation_history_patient,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )
            response_patient, _ = chat_text(patient_model_name, conversation_history_patient)
            # print("patient:", response_patient)

            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                            "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                            "content":response_patient})

            # Doctor talks
            # response_doctor = doctor_model(conversation_history_doctor)
            # response_doctor = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=conversation_history_doctor,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )
            response_doctor, _ = chat_text(doctor_model_name, conversation_history_doctor)

            # print("doctor:", response_doctor)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                            "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})
            

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break
        
        # print("now here")

        if flag == 0:
            # multi-turn conversation + MCQ
            # print("new here1")
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            # multiturn_mcq = doctor_model(multiturn_convo_without_diagnosis)
            # multiturn_mcq = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=multiturn_convo_without_diagnosis,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )
            multiturn_mcq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq

            # print("new here2")
            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            # multiturn_frq = doctor_model(multiturn_convo_without_diagnosis)
            # multiturn_frq = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=multiturn_convo_without_diagnosis,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )
            multiturn_frq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq
            diagnosis = conversation_history_doctor[-1]["content"]

            llm_as_a_judge_prompt = """
            Return either 'True' or 'False' for whether or not the following diagnosis is equivalent with the ground truth diagnosis. If one is the scientific name for the other, they are still equivalent. 
            
            Ground Truth Diagnosis: {case_gt_answer}.

            Diagnosis: {diagnosis}.

            Only respond with one word: 'True' or 'False'
            """.format(case_gt_answer=case_gt_answer, diagnosis= diagnosis)

            # print("here1.2")
            # print(llm_as_a_judge_prompt)


            check_success, _= chat_text("gpt-4o-mini-2024-07-18", [{"role": "system","content": llm_as_a_judge_prompt}], temperature=0.0)

            # print("check_success:", check_success)

            success_pre_intervention = check_success.lower() == "true"
            # print("success bool:", success)
            stats[f"trial_{j}"]["ground_truth"] = case_gt_answer
            stats[f"trial_{j}"]["reward"] = success_pre_intervention

            #print("new here3")
            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            # singleturn_mcq = doctor_model(singleturn_convo_without_diagnosis)
            # singleturn_mcq = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=singleturn_convo_without_diagnosis,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )
            singleturn_mcq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq

            # print("new here4")
            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            # singleturn_frq = doctor_model(singleturn_convo_without_diagnosis)
            # singleturn_frq = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=singleturn_convo_without_diagnosis,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )
            singleturn_frq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            

            # print("new here5")
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses, doctor_model_name)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation


            # print("new here6")
            # summarized conversation + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            convo = [{"role": "user", "content": summarized_mcq_prompt}]
            # summarized_mcq_ans = doctor_model(convo)
            # summarized_mcq_ans = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=convo,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )

            # print("new here7")
            summarized_mcq_ans, _ = chat_text(doctor_model_name, convo)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation + FRQ

            # print("new here8")
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            convo = [{"role": "user", "content": summarized_frq_prompt}]
            # summarized_frq_ans = doctor_model(convo)
            # summarized_frq_ans = openai.chat.completions.create(
            #     model=doctor_model_name,
            #     messages=convo,
            #     max_completion_tokens=4096,
            #     temperature = 0.1
            # )

            # print("new here9")
            summarized_frq_ans, _ = chat_text(doctor_model_name, convo)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
            
            j += 1
        # print("new here10")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            with open(save_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []
        if isinstance(existing, dict):
            existing = [existing]
        elif not isinstance(existing, list):
            existing = []

        existing.append(stats)
        with open(save_path, "w") as f:
            json.dump(existing, f, indent=2)
        # json.dump(stats, open(save_path, 'w'))


        if (success_pre_intervention):
            print(f"✅ {case_id}, succesful")
        else:
            print(f"❌ {case_id} unsuccesful")




        #run interventions now

        if (not intervention):
            return success_pre_intervention, False
        
        if (success_pre_intervention):

            #add filler entry to intervened json file
            intervened_stats = {}
            intervened_stats["case_id"] = case_id
            intervened_stats["case_vignette"] = case_desc
            intervened_stats["question"] = question
            intervened_stats["reward"] = True
            intervened_stats["intervened_index"] = -1
            intervened_stats["intervened_message"] = "no intervention was needed. already passed"
            intervened_stats["success_prev"] = success_pre_intervention
            if os.path.exists(intervened_path) and os.path.getsize(intervened_path) > 0:
                with open(intervened_path, "r") as f:
                    try:
                        existing = json.load(f)
                    except json.JSONDecodeError:
                        existing = []
            else:
                existing = []
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []

            existing.append(intervened_stats)
            with open(intervened_path, "w") as f:
                json.dump(existing, f, indent=2)

            print(f"✅ {case_id} already succesful")
            return success_pre_intervention, True
        
        print("beginning intervention on case:", case_id)
        answer_list, intervening_agent_conversation = run_intervention(stats,N=5, temperature=1.0)
        # print("answer list:",answer_list)
        if (answer_list == None or answer_list[0] == False):
            intervened_stats = {}
            intervened_stats["case_id"] = case_id
            intervened_stats["case_vignette"] = case_desc
            intervened_stats["question"] = question
            intervened_stats["reward"] = True
            intervened_stats["intervened_index"] = -1
            intervened_stats["intervened_message"] = "intervening agent failed."
            intervened_stats["success_prev"] = success_pre_intervention
            if os.path.exists(intervened_path) and os.path.getsize(intervened_path) > 0:
                with open(intervened_path, "r") as f:
                    try:
                        existing = json.load(f)
                    except json.JSONDecodeError:
                        existing = []
            else:
                existing = []
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []

            existing.append(intervened_stats)
            with open(intervened_path, "w") as f:
                json.dump(existing, f, indent=2)
            return success_pre_intervention, False
            pass
            continue
            # add to intervention file 

        current_conv_histories = {"task_id":case_id,"traj":intervening_agent_conversation}
        
        #add conversation history to running file
        if os.path.exists(conv_save_path) and os.path.getsize(conv_save_path) > 0:
            with open(conv_save_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []
        if isinstance(existing, dict):
            existing = [existing]
        elif not isinstance(existing, list):
            existing = []

        existing.append(current_conv_histories)
        with open(conv_save_path, "w") as f:
            json.dump(existing, f, indent=2)

        def add_intervention(trajectory, intervention_text, intervention_id):
            try:
                if type(intervention_id) == int:
                    idx_intervention =  intervention_id  
                else:
                    idx_b = -1 if (intervention_id.rfind("B") == -1) else intervention_id.rfind("B")
                    idx_intervention = int(intervention_id[idx_b+1:])
            except Exception as e:
                print(f"first converting to int idx: {case_id} error: {e}.")
                idx_intervention = 1

            new_trajectory = trajectory[:min(idx_intervention+1, len(trajectory) - 1)]
            if (len(trajectory) - 1 < idx_intervention+1):
                print(f"was out of range task id: {case_id}")
            new_trajectory.append(
                {
                    "role":"system",
                    "content": "[*INTERVENTION*: " +intervention_text + "]"
                }
            )

            return new_trajectory        

        for best_of_n_iterator, intervention in enumerate(answer_list):
            try:

                intervention_id = intervention["id"]
                if type(intervention_id) == int:
                    idx_intervention =  intervention_id  
                else:
                    idx_b = -1 if (intervention_id.rfind("B") == -1) else intervention_id.rfind("B")
                    if idx_b == -1:
                        idx_b = -1 if (intervention_id.rfind("A") == -1) else intervention_id.rfind("A")
                    idx_intervention = int(intervention_id[idx_b+1:])

            except Exception as e:
                print(f"converting to int idx: {case_id} error: {e}.")
                idx_intervention = 1

            answer_list[best_of_n_iterator]["id"] = idx_intervention

        for best_of_n_iterator, intervention in enumerate(answer_list):
            try:

                failure_id = intervention["failure_id"]
                if type(failure_id) == int:
                    idx_failure =  failure_id  
                else:
                    idx_b = -1 if (failure_id.rfind("B") == -1) else failure_id.rfind("B")
                    if idx_b == -1:
                        idx_b = -1 if (failure_id.rfind("A") == -1) else failure_id.rfind("A")
                    idx_failure = int(failure_id[idx_b+1:])

            except Exception as e:
                print(f"failure id; converting to int idx: {case_id} error: {e}.")
                idx_failure = 1

            answer_list[best_of_n_iterator]["failure_id"] = idx_failure

        sorted_answer_list = sorted(answer_list, key=lambda x: x['id']) 

        did_improve = False

        for best_of_n_iterator, intervention in enumerate(sorted_answer_list):
            intervened_stats = {}
            intervened_stats["case_id"] = case_id
            intervened_stats["case_vignette"] = case_desc
            intervened_stats["question"] = question
            print(f"trying out task id={case_id}, intervention {best_of_n_iterator}")
            
            failure_brief = intervention["failure_brief"]
            failure_id = intervention["failure_id"]
            intervention_text = intervention["intervention_text"]
            intervention_id = intervention["id"]
            
            



            print(f"intervention id: {intervention_id} intervention txt: {intervention_text}")
            
            #add intervention to trajectory
            new_intervened_trajectory = add_intervention(stats["trial_0"]["multiturn_conversation"], intervention_text, intervention_id)


            k = 0
            while k < num_runs:
                intervened_stats["improved"] = False
                intervened_stats["reward"] = False
                intervened_stats["intervened_first_or_last"] = best_of_n_iterator
                intervened_stats["failure_brief"] = failure_brief
                intervened_stats["failure_index"] = failure_id
                intervened_stats["intervened_message"] = intervention_text
                intervened_stats["success_prev"] = success_pre_intervention
                intervened_stats["intervened_index"] = intervention_id
                intervened_stats[f"trial_{k}"] = {}
                flag = 0
                intervened_stats[f"trial_{k}"]["vignette_mcq"] = stats[f"trial_0"]["vignette_mcq"]
                
                # Vignette + FRQ
                intervened_stats[f"trial_{k}"]["vignette_frq"] = stats[f"trial_0"]["vignette_frq"]
                
                # Multi-turn conversation experiments
                conversation_history_doctor = copy.deepcopy(new_intervened_trajectory)
                conversation_history_patient = copy.deepcopy(new_intervened_trajectory)

                # for message_idx in range(len(conversation_history_patient)):
                #     if (conversation_history_patient[message_idx]["role"]) == "user":
                #         conversation_history_patient[message_idx]["role"] = "temp"
                # for message_idx in range(len(conversation_history_patient)):
                #     if (conversation_history_patient[message_idx]["role"]) == "assistant":
                #         conversation_history_patient[message_idx]["role"] = "user"
                # for message_idx in range(len(conversation_history_patient)):
                #     if (conversation_history_patient[message_idx]["role"]) == "temp":
                #         conversation_history_patient[message_idx]["role"] = "assistant"
                conversation_history_patient[0]["content"] = patient_prompt
                # print("patient convo history:")
                # pprint.pprint(conversation_history_patient)
                
                # print("doctor convo history:")
                # pprint.pprint(conversation_history_doctor)
                # print(f"here1, {case_id}, interventiontion id: {intervention_id}")
                while True:
                    response_patient, _ = chat_text(patient_model_name, conversation_history_patient)

                    if response_patient is None:
                        flag=1
                        break
                    # print(f"patient: {response_patient}")
                    conversation_history_doctor.append({"role":"user",
                                                    "content":response_patient})
                    conversation_history_patient.append({"role":"assistant",
                                                    "content":response_patient})

                    response_doctor, _ = chat_text(doctor_model_name, conversation_history_doctor)
                    # print(f"doctor: {response_doctor}")
                    if response_doctor is None:
                        flag=1
                        break

                    conversation_history_doctor.append({"role":"assistant",
                                                    "content": response_doctor})
                    conversation_history_patient.append({"role":"user", 
                                                        "content": response_doctor})
                    
                    if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                        intervened_stats[f"trial_{k}"]["multiturn_conversation"] = conversation_history_doctor
                        break
                # print(f"made it out,here2 {case_id}, interventiontion id: {intervention_id}")
                if flag == 0:

                    prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
                    multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
                    multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})

                    multiturn_mcq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["multiturn_mcq"] = multiturn_mcq

                    # print("new here2")
                    # multi-turn conversation + FRQ
                    prompt = get_frq_after_conversation_prompt(question)
                    multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
                    multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
                    multiturn_frq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["multiturn_frq"] = multiturn_frq
                    
                    diagnosis = conversation_history_doctor[-1]["content"]
                    intervened_stats[f"trial_{k}"]["diagnosis"] = multiturn_frq
                    llm_as_a_judge_prompt = """
                    Return either 'True' or 'False' for whether or not the following diagnosis is equivalent with the ground truth diagnosis. If one is the scientific name for the other, they are still equivalent. 
                    
                    Ground Truth Diagnosis: {case_gt_answer}.

                    Diagnosis: {diagnosis}.

                    Only respond with one word: 'True' or 'False'
                    """.format(case_gt_answer=case_gt_answer, diagnosis= diagnosis)

                    check_success, _= chat_text("gpt-4o-mini-2024-07-18", [{"role": "system","content": llm_as_a_judge_prompt}], temperature=0.0)
                    success_post_intervention = check_success.lower() == "true"
                    intervened_stats["success_after"] = success_post_intervention
                    if success_post_intervention:
                        intervened_stats["reward"] = True
                        # intervened_stats["success_after"] = True

                    if (success_post_intervention == True and success_pre_intervention == False):
                        print(f"***IMPROVED*** case_id: {case_id} at {intervention_id}")
                        intervened_stats["improved"] = True
                        did_improve = True
                        
                    # print("success bool:", success)
                    intervened_stats[f"trial_{k}"]["ground_truth"] = case_gt_answer
                    intervened_stats[f"trial_{k}"]["reward"] = success_post_intervention


                    prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
                    singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
                    singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})

                    singleturn_mcq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["singleturn_mcq"] = singleturn_mcq

                    prompt = get_frq_after_conversation_prompt(question)
                    singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
                    singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})

                    singleturn_frq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["singleturn_frq"] = singleturn_frq

                    pat_responses = get_patient_responses(conversation_history_doctor[1:])
                    summarized_conversation = convert_to_summarized(pat_responses, doctor_model_name)
                    intervened_stats[f"trial_{k}"]["summarized_conversation"] = summarized_conversation

                    summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
                    convo = [{"role": "user", "content": summarized_mcq_prompt}]

                    summarized_mcq_ans, _ = chat_text(doctor_model_name, convo)
                    intervened_stats[f"trial_{k}"]["summarized_mcq"] = summarized_mcq_ans

                    # summarized conversation + FRQ
                    summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
                    convo = [{"role": "user", "content": summarized_frq_prompt}]

                    summarized_frq_ans, _ = chat_text(doctor_model_name, convo)
                    intervened_stats[f"trial_{k}"]["summarized_frq"] = summarized_frq_ans
                    
                    k += 1
                
                # print(f"made it out, here3 {case_id}, interventiontion id: {intervention_id}")
                
                
                if (success_post_intervention):
                    print(f"✅ {case_id} w/ intervention @ {intervention_id}, succesful")
                else:
                    print(f"❌ {case_id} w/ intervention @ {intervention_id}, unsuccesful")
                # print("new here10")

                if os.path.exists(intervened_path) and os.path.getsize(intervened_path) > 0:
                    with open(intervened_path, "r") as f:
                        try:
                            existing = json.load(f)
                        except json.JSONDecodeError:
                            existing = []
                else:
                    existing = []
                if isinstance(existing, dict):
                    existing = [existing]
                elif not isinstance(existing, list):
                    existing = []

                existing.append(intervened_stats)
                with open(intervened_path, "w") as f:
                    json.dump(existing, f, indent=2)
                k+=1
        
        
        # if (did_improve):
        #     improved_count += 1
        return success_pre_intervention, success_post_intervention
    # except Exception as e:
    #     print(f"caseid: {case_id} error: {e}")
    #     return False, False



        


#two important ones:
def craftmd_gpt_baseline(case, path_dir, doctor_model_name, patient_model_name, intervention, num_runs = 1):
    case_id, case_desc, specialty, mcq_choices, case_gt_answer = case
    case_desc, question = get_case_without_question(case_desc)
    # try:

    print(f'Thread for case id: {case_id} dispatched.')

    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt(specialty)
    patient_prompt = get_patient_prompt(case_desc)
    stats = {}  
    j = 0

    save_path = f'{path_dir}/transcript.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
    else:
        os.makedirs(path_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({}, f)

    stats = {}

    stats["case_id"] = case_id
    stats["case_vignette"] = case_desc
    stats["question"] = question
    stats["category"] = specialty
    




    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt(specialty, case_desc, question, mcq_choices)
        convo = [{"role": "system", "content": vignette_mcq_prompt}]
        vignette_mcq_ans, _ = chat_text(doctor_model_name, convo)

        stats[f"trial_{j}"]["vignette_mcq"] = vignette_mcq_ans
        
        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt(specialty, case_desc, question)
        convo = [{"role": "system", "content": vignette_frq_prompt}]
        vignette_frq_ans, _ = chat_text(doctor_model_name, convo)

        # print("here5")
        stats[f"trial_{j}"]["vignette_frq"] = vignette_frq_ans
        
        # Multi-turn conversation experiments
        conversation_history_doctor = [{"role": "system", "content": doctor_prompt},
                                    {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]

        # print("beginning conversation")
        while True:
            # Patient talks
            response_patient, _ = chat_text(patient_model_name, conversation_history_patient)
            # print("patient:", response_patient)

            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                            "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                            "content":response_patient})

            # Doctor talks
            response_doctor, _ = chat_text(doctor_model_name, conversation_history_doctor)

            # print("doctor:", response_doctor)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                            "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})
            

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break
        
        # print("now here")

        if flag == 0:
            # multi-turn conversation + MCQ
            # print("new here1")
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_mcq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq

            # print("new here2")
            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_frq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq
            diagnosis = conversation_history_doctor[-1]["content"]

            llm_as_a_judge_prompt = """
            Return either 'True' or 'False' for whether or not the following diagnosis is equivalent with the ground truth diagnosis. If one is the scientific name for the other, they are still equivalent. 
            
            Ground Truth Diagnosis: {case_gt_answer}.

            Diagnosis: {diagnosis}.

            Only respond with one word: 'True' or 'False'
            """.format(case_gt_answer=case_gt_answer, diagnosis= diagnosis)

            # print("here1.2")
            # print(llm_as_a_judge_prompt)


            check_success, _= chat_text("gpt-4o-mini-2024-07-18", [{"role": "system","content": llm_as_a_judge_prompt}], temperature=0.0)

            # print("check_success:", check_success)

            success_pre_intervention = check_success.lower() == "true"
            # print("success bool:", success)
            stats[f"trial_{j}"]["ground_truth"] = case_gt_answer
            stats[f"trial_{j}"]["reward"] = success_pre_intervention

            #print("new here3")
            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_mcq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq

            # print("new here4")
            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_frq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            

            # print("new here5")
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses, doctor_model_name)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation


            # print("new here6")
            # summarized conversation + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            convo = [{"role": "user", "content": summarized_mcq_prompt}]

            # print("new here7")
            summarized_mcq_ans, _ = chat_text(doctor_model_name, convo)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation + FRQ

            # print("new here8")
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            convo = [{"role": "user", "content": summarized_frq_prompt}]

            # print("new here9")
            summarized_frq_ans, _ = chat_text(doctor_model_name, convo)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
            
            j += 1
        # print("new here10")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            with open(save_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []
        if isinstance(existing, dict):
            existing = [existing]
        elif not isinstance(existing, list):
            existing = []

        existing.append(stats)
        
        with open(save_path, "w") as f:
            json.dump(existing, f, indent=2)
        # json.dump(stats, open(save_path, 'w'))


        if (success_pre_intervention):
            print(f"✅ {case_id}, succesful")
        else:
            print(f"❌ {case_id} unsuccesful")

def craftmd_gpt_intervention(case, baseline_trajectory, path_dir, doctor_model_name, patient_model_name, intervening_model):
    if case == None:
        return False
    

    case_id, case_desc, specialty, mcq_choices, case_gt_answer = case
    case_desc, question = get_case_without_question(case_desc)
    doctor_prompt = get_doctor_prompt(specialty)
    patient_prompt = get_patient_prompt(case_desc)
    try:
        print(f'Thread for case id: {case_id} dispatched.')
        
        intervened_path = os.path.join(path_dir,"intervened-transcripts.json")
        conv_save_path = os.path.join(path_dir,"agent_conversation_histories.json")

        if not os.path.exists(intervened_path):
            os.makedirs(path_dir, exist_ok=True)
            with open(intervened_path, "w") as f:
                json.dump({}, f)
        
        if not os.path.exists(conv_save_path):
            os.makedirs(path_dir, exist_ok=True)
            with open(conv_save_path, "w") as f:
                json.dump({}, f)

        intervened_stats = {}
        success_pre_intervention = baseline_trajectory["trial_0"]["reward"]
        if (baseline_trajectory["trial_0"]["reward"]):
            #add filler entry to intervened json file
            intervened_stats = {}
            intervened_stats["case_id"] = case_id
            intervened_stats["case_vignette"] = case_desc
            intervened_stats["question"] = question
            intervened_stats["reward"] = True
            intervened_stats["intervened_index"] = -1
            intervened_stats["intervened_message"] = "no intervention was needed. already passed"
            intervened_stats["success_prev"] = True
            if os.path.exists(intervened_path) and os.path.getsize(intervened_path) > 0:
                with open(intervened_path, "r") as f:
                    try:
                        existing = json.load(f)
                    except json.JSONDecodeError:
                        existing = []
            else:
                existing = []
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []

            existing.append(intervened_stats)
            with open(intervened_path, "w") as f:
                json.dump(existing, f, indent=2)

            print(f"✅ {case_id} already succesful")
            return True
        
        





            # return False
        

        print("beginning intervention on case:", case_id)
        answer_list, intervening_agent_conversation = run_intervention(baseline_trajectory,N=3, temperature=0.1, model=intervening_model)

        print("answer list:",answer_list)
        # intervenor agent fails
        if (answer_list == None or answer_list[0] == False):
            intervened_stats = {}
            intervened_stats["case_id"] = case_id
            intervened_stats["case_vignette"] = case_desc
            intervened_stats["question"] = question
            intervened_stats["reward"] = True
            intervened_stats["intervened_index"] = -1
            intervened_stats["intervened_message"] = "intervening agent failed."
            intervened_stats["success_prev"] = success_pre_intervention
            if os.path.exists(intervened_path) and os.path.getsize(intervened_path) > 0:
                with open(intervened_path, "r") as f:
                    try:
                        existing = json.load(f)
                    except json.JSONDecodeError:
                        existing = []
            else:
                existing = []
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []

            existing.append(intervened_stats)
            with open(intervened_path, "w") as f:
                json.dump(existing, f, indent=2)
            return success_pre_intervention, False
            # add to intervention file 

        current_conv_histories = {"task_id":case_id,"traj":intervening_agent_conversation}
        
        #add conversation history to running file
        if os.path.exists(conv_save_path) and os.path.getsize(conv_save_path) > 0:
            with open(conv_save_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []
        if isinstance(existing, dict):
            existing = [existing]
        elif not isinstance(existing, list):
            existing = []

        existing.append(current_conv_histories)
        with open(conv_save_path, "w") as f:
            json.dump(existing, f, indent=2)

        def add_intervention(trajectory, intervention_text, intervention_id):
            try:
                if type(intervention_id) == int:
                    idx_intervention =  intervention_id  
                else:
                    idx_b = -1 if (intervention_id.rfind("B") == -1) else intervention_id.rfind("B")
                    idx_intervention = int(intervention_id[idx_b+1:])
            except Exception as e:
                print(f"first converting to int idx: {case_id} error: {e}.")
                idx_intervention = 1

            new_trajectory = trajectory[:min(idx_intervention+1, len(trajectory) - 1)]
            if (len(trajectory) - 1 < idx_intervention+1):
                print(f"was out of range task id: {case_id}")
            new_trajectory.append(
                {
                    "role":"system",
                    "content": "[*INTERVENTION*: " +intervention_text + "]"
                }
            )

            return new_trajectory        

        for best_of_n_iterator, intervention in enumerate(answer_list):
            try:

                intervention_id = intervention["id"]
                if type(intervention_id) == int:
                    idx_intervention =  intervention_id  
                else:
                    idx_b = -1 if (intervention_id.rfind("B") == -1) else intervention_id.rfind("B")
                    if idx_b == -1:
                        idx_b = -1 if (intervention_id.rfind("A") == -1) else intervention_id.rfind("A")
                    idx_intervention = int(intervention_id[idx_b+1:])

            except Exception as e:
                print(f"converting to int idx: {case_id} error: {e}.")
                idx_intervention = 1

            answer_list[best_of_n_iterator]["id"] = idx_intervention

        for best_of_n_iterator, intervention in enumerate(answer_list):
            try:

                failure_id = intervention["failure_id"]
                if type(failure_id) == int:
                    idx_failure =  failure_id  
                else:
                    idx_b = -1 if (failure_id.rfind("B") == -1) else failure_id.rfind("B")
                    if idx_b == -1:
                        idx_b = -1 if (failure_id.rfind("A") == -1) else failure_id.rfind("A")
                    idx_failure = int(failure_id[idx_b+1:])

            except Exception as e:
                print(f"failure id; converting to int idx: {case_id} error: {e}.")
                idx_failure = 1

            answer_list[best_of_n_iterator]["failure_id"] = idx_failure

        sorted_answer_list = sorted(answer_list, key=lambda x: x['id']) 

        did_improve = False

        for best_of_n_iterator, intervention in enumerate(sorted_answer_list):
            intervened_stats = {}
            intervened_stats["case_id"] = case_id
            intervened_stats["case_vignette"] = case_desc
            intervened_stats["question"] = question
            print(f"trying out task id={case_id}, intervention {best_of_n_iterator}")
            
            failure_brief = intervention["failure_brief"]
            failure_id = intervention["failure_id"]
            intervention_text = intervention["intervention_text"]
            intervention_id = intervention["id"]
            
            



            print(f"intervention id: {intervention_id} intervention txt: {intervention_text}")
            
            #add intervention to trajectory
            new_intervened_trajectory = add_intervention(baseline_trajectory["trial_0"]["multiturn_conversation"], intervention_text, intervention_id)


            k = 0
            while k < 1:
                intervened_stats["improved"] = False
                intervened_stats["reward"] = False
                intervened_stats["intervened_first_or_last"] = best_of_n_iterator
                intervened_stats["failure_brief"] = failure_brief
                intervened_stats["failure_index"] = failure_id
                intervened_stats["intervened_message"] = intervention_text
                intervened_stats["success_prev"] = success_pre_intervention
                intervened_stats["intervened_index"] = intervention_id
                intervened_stats[f"trial_{k}"] = {}
                flag = 0
                intervened_stats[f"trial_{k}"]["vignette_mcq"] = baseline_trajectory[f"trial_0"]["vignette_mcq"]
                
                # Vignette + FRQ
                intervened_stats[f"trial_{k}"]["vignette_frq"] = baseline_trajectory[f"trial_0"]["vignette_frq"]
                
                # Multi-turn conversation experiments
                conversation_history_doctor = copy.deepcopy(new_intervened_trajectory)
                conversation_history_patient = copy.deepcopy(new_intervened_trajectory)

                conversation_history_patient[0]["content"] = patient_prompt
                # print("patient convo history:")
                # pprint.pprint(conversation_history_patient)
                
                # print("doctor convo history:")
                # pprint.pprint(conversation_history_doctor)
                # print(f"here1, {case_id}, interventiontion id: {intervention_id}")
                while True:
                    response_patient, _ = chat_text(patient_model_name, conversation_history_patient)

                    if response_patient is None:
                        flag=1
                        break
                    # print(f"patient: {response_patient}")
                    conversation_history_doctor.append({"role":"user",
                                                    "content":response_patient})
                    conversation_history_patient.append({"role":"assistant",
                                                    "content":response_patient})

                    response_doctor, _ = chat_text(doctor_model_name, conversation_history_doctor)
                    # print(f"doctor: {response_doctor}")
                    if response_doctor is None:
                        flag=1
                        break

                    conversation_history_doctor.append({"role":"assistant",
                                                    "content": response_doctor})
                    conversation_history_patient.append({"role":"user", 
                                                        "content": response_doctor})
                    
                    if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                        intervened_stats[f"trial_{k}"]["multiturn_conversation"] = conversation_history_doctor
                        break
                print(f"made it out,here2 {case_id}, interventiontion id: {intervention_id}")
                if flag == 0:

                    prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
                    multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
                    multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})

                    multiturn_mcq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["multiturn_mcq"] = multiturn_mcq

                    # print("new here2")
                    # multi-turn conversation + FRQ
                    prompt = get_frq_after_conversation_prompt(question)
                    multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
                    multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
                    multiturn_frq, _ = chat_text(doctor_model_name, multiturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["multiturn_frq"] = multiturn_frq
                    
                    diagnosis = conversation_history_doctor[-1]["content"]
                    intervened_stats[f"trial_{k}"]["diagnosis"] = multiturn_frq
                    llm_as_a_judge_prompt = """
                    Return either 'True' or 'False' for whether or not the following diagnosis is equivalent with the ground truth diagnosis. If one is the scientific name for the other, they are still equivalent. 
                    
                    Ground Truth Diagnosis: {case_gt_answer}.

                    Diagnosis: {diagnosis}.

                    Only respond with one word: 'True' or 'False'
                    """.format(case_gt_answer=case_gt_answer, diagnosis= diagnosis)

                    check_success, _= chat_text("gpt-4o-mini-2024-07-18", [{"role": "system","content": llm_as_a_judge_prompt}], temperature=0.0)
                    success_post_intervention = check_success.lower() == "true"
                    intervened_stats["success_after"] = success_post_intervention
                    if success_post_intervention:
                        intervened_stats["reward"] = True
                        # intervened_stats["success_after"] = True

                    if (success_post_intervention == True and success_pre_intervention == False):
                        print(f"***IMPROVED*** case_id: {case_id} at {intervention_id}")
                        intervened_stats["improved"] = True
                        did_improve = True
                        
                    # print("success bool:", success)
                    intervened_stats[f"trial_{k}"]["ground_truth"] = case_gt_answer
                    intervened_stats[f"trial_{k}"]["reward"] = success_post_intervention


                    prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
                    singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
                    singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})

                    singleturn_mcq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["singleturn_mcq"] = singleturn_mcq

                    prompt = get_frq_after_conversation_prompt(question)
                    singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
                    singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})

                    singleturn_frq, _ = chat_text(doctor_model_name, singleturn_convo_without_diagnosis)
                    intervened_stats[f"trial_{k}"]["singleturn_frq"] = singleturn_frq

                    pat_responses = get_patient_responses(conversation_history_doctor[1:])
                    summarized_conversation = convert_to_summarized(pat_responses, doctor_model_name)
                    intervened_stats[f"trial_{k}"]["summarized_conversation"] = summarized_conversation

                    summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
                    convo = [{"role": "user", "content": summarized_mcq_prompt}]

                    summarized_mcq_ans, _ = chat_text(doctor_model_name, convo)
                    intervened_stats[f"trial_{k}"]["summarized_mcq"] = summarized_mcq_ans

                    # summarized conversation + FRQ
                    summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
                    convo = [{"role": "user", "content": summarized_frq_prompt}]

                    summarized_frq_ans, _ = chat_text(doctor_model_name, convo)
                    intervened_stats[f"trial_{k}"]["summarized_frq"] = summarized_frq_ans
                    
                    k += 1
                
                # print(f"made it out, here3 {case_id}, interventiontion id: {intervention_id}")
                
                
                if (success_post_intervention):
                    print(f"✅ {case_id} w/ intervention @ {intervention_id}, succesful")
                else:
                    print(f"❌ {case_id} w/ intervention @ {intervention_id}, unsuccesful")
                # print("new here10")

                if os.path.exists(intervened_path) and os.path.getsize(intervened_path) > 0:
                    with open(intervened_path, "r") as f:
                        try:
                            existing = json.load(f)
                        except json.JSONDecodeError:
                            existing = []
                else:
                    existing = []
                if isinstance(existing, dict):
                    existing = [existing]
                elif not isinstance(existing, list):
                    existing = []

                existing.append(intervened_stats)
                with open(intervened_path, "w") as f:
                    json.dump(existing, f, indent=2)
                k+=1
        
        
        # if (did_improve):
        #     improved_count += 1
        return success_pre_intervention, success_post_intervention
    except Exception as e:
        print(f"caseid: {case_id} error: {e}")
        return False







def craftmd_opensource(case, path_dir, doctor_model, doctor_tokenizer, num_runs = 5):

    case_id, case_desc, specialty, mcq_choices = case
    case_desc, question = get_case_without_question(case_desc)

    print(f'Thread for case id: {case_id} dispatched.')
    
    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt(specialty)
    patient_prompt = get_patient_prompt(case_desc)

    stats = {}
    j = 0

    save_path = f'{path_dir}/{case_id}.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
            
    if j == num_runs-1:
        return

    stats["case_vignette"] = case_desc
    stats["question"] = question
    
    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt(specialty, case_desc, question, mcq_choices)
        conv = [{"role": "user", "content": vignette_mcq_prompt}]
        vignette_mcq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
        stats[f"trial_{j}"]["vignette_mcq"] = vignette_mcq_ans

        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt(specialty, case_desc, question)
        conv = [{"role": "user", "content": vignette_frq_prompt}]
        vignette_frq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
        stats[f"trial_{j}"]["vignette_frq"] = vignette_frq_ans

        # Multi-turn conversation experiments
        conversation_history_doctor = [{"role": "user", "content": doctor_prompt},
                                       {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]

        while True:
            # Patient talks
            response_patient = patient_model(conversation_history_patient)
            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})


            # Doctor talks
            response_doctor = call_open_llm(doctor_model, doctor_tokenizer, conversation_history_doctor)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user",
                                                "content": response_doctor})


            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor) or (len(conversation_history_doctor)>=30):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break

        if flag==0:
            # multi-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:-1])
            multiturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            multiturn_mcq = call_open_llm(doctor_model, doctor_tokenizer, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq
    
            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:-1])
            multiturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            multiturn_frq = call_open_llm(doctor_model, doctor_tokenizer, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq


    
            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:3])
            singleturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            singleturn_mcq = call_open_llm(doctor_model, doctor_tokenizer, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq
    
            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:3])
            singleturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            singleturn_frq = call_open_llm(doctor_model, doctor_tokenizer, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation

            # summarized conversation (with physical exam) + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            conv = [{"role": "user", "content": summarized_mcq_prompt}]
            summarized_mcq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation (with physical exam) + FRQ
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            conv = [{"role": "user", "content": summarized_frq_prompt}]
            summarized_frq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
    
            j += 1

        json.dump(stats, open(save_path, 'w'))
        
        
        
def craftmd_multimodal(case, img_dir, path_dir, deployment_name, num_runs = 5):

    case_id, case_desc, mcq_choices = case
    case_desc, question = get_case_without_question(case_desc)
    
    image_path = f'{img_dir}/{case_id.split("_")[1]}.jpeg'
    data_url = local_image_to_data_url(image_path)

    print(f'Thread for case id: {case_id} dispatched.')
    
    doctor_model = call_gpt4v_api
    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt_multimodal()
    patient_prompt = get_patient_prompt(case_desc)

    stats = {}
    j = 0

    save_path = f'{path_dir}/{case_id}.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
            
    if j == num_runs-1:
        return

    stats["case_vignette"] = case_desc
    stats["question"] = question
    
    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt_multimodal(case_desc, question, mcq_choices)
        conv = [{"role": "system", "content": [{"type": "text",
                                                "text": vignette_mcq_prompt},
                                               {"type": "image_url",
                                                "image_url": {"url":data_url}}]}]
        vignette_mcq_ans = doctor_model(conv, deployment_name)
        
        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt_multimodal(case_desc, question)
        conv = [{"role": "system", "content": [{"type": "text",
                                                "text": vignette_frq_prompt},
                                               {"type": "image_url",
                                                "image_url": {"url":data_url}}]}]
        vignette_frq_ans = doctor_model(conv, deployment_name)
        
        # Conversation experiments
        conversation_history_doctor = [{"role": "system", "content": [{"type": "text",
                                                                       "text": doctor_prompt},
                                                                      {"type": "image_url",
                                                                       "image_url": {"url": data_url}}]},
                                       {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]
        while True:
            # Patient talks
            response_patient = patient_model(conversation_history_patient) 
            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})

            # Doctor talks
            response_doctor = doctor_model(conversation_history_doctor, deployment_name)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})
            

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break
        
        if flag == 0:
            # multi-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_mcq = doctor_model(multiturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq

            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_frq = doctor_model(multiturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq

            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_mcq = doctor_model(singleturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq

            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_frq = doctor_model(singleturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation

            # summarized conversation + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            convo = [{"role": "user", "content": summarized_mcq_prompt}]
            summarized_mcq_ans = doctor_model(convo, deployment_name)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation + FRQ
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            convo = [{"role": "user", "content": summarized_frq_prompt}]
            summarized_frq_ans = doctor_model(convo, deployment_name)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
            
            j += 1
        
        json.dump(stats, open(save_path, 'w'))
        