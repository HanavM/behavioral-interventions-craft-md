##### MELD Analysis
def get_meld_prompt(first_half, word_count):
    system_prompt = "You are given the first half of a medical case vignette."
    system_prompt += " Generate the second half of the case vignette. You do not have to give the diagnosis."
    system_prompt += f" Generate only {word_count} words."
    system_prompt += f"\n**Case Vignette**: {first_half}"
    return system_prompt


##### CRAFT-MD prompts
def get_vignette_mcq_prompt(specialty, case_vignette, question, choices):
    if specialty=="Other":
        system_prompt = "You are an AI doctor."
    else:
        system_prompt = f"You are an AI doctor specializing in {specialty}."
        
    system_prompt += " You are given the patient's symptoms and a list of possible answer choices."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Select the correct choice, and give the answer as a short response. Do not explain."
    system_prompt += f"\n**Symptoms**: {case_vignette} {question}"
    system_prompt += f"\n**Choices**: {choices}"
    return system_prompt

def get_vignette_frq_prompt(specialty, case_vignette, question):
    if specialty=="Other":
        system_prompt = "You are an AI doctor."
    else:
        system_prompt = f"You are an AI doctor specializing in {specialty}."
        
    system_prompt += " Based on the given patient symptoms, give the correct answer as a short response. Do not explain."
    system_prompt += f"\n**Symptoms**: {case_vignette} {question}"
    return system_prompt

def get_doctor_prompt(specialty):
    if specialty=="Other":
        system_prompt = "You are an AI doctor."
    else:
        system_prompt = f"You are an AI doctor specializing in {specialty}."
    system_prompt += " Arrive at a diagnosis of a patient's medical condition."
    system_prompt += " Ask only one question at a time, and it should not be more than 1 line."
    system_prompt += " Continue asking questions until you're 100% confident of the diagnosis."
    system_prompt += " Do not ask the same question multiple times. Ask different questions to cover more information."
    system_prompt += " The questions should cover age and sex of the patient, current symptoms, medical history of illness and medications, and relevant family history if necessary."
    system_prompt += " Keep your questions short and brief to not confuse the patient. "
    system_prompt += " After you're done asking questions, give the final diagnosis as a short response. Do not explain, only give the diagnosis name."
    system_prompt += " You must state '**Final Diagnosis:**' at the beginning of your response, otherwise you will be penalized."
    system_prompt += " You must give only 1 diagnosis otherwise you will be penalized. Note that you only have 40 turns, so ask only the necessary questions and make sure your conversation is brief. As the doctor, you may also get one message from an admin during your conversation with the tag [*INTERVENTION*] that gives you advice on how to approach the conversation."
    return system_prompt

def get_patient_prompt(case_desc): 
    system_prompt = "You are a patient."
    system_prompt += " You do not have any medical knowledge."
    system_prompt += " You have to describe your symptoms from the given case vignette based on the questions asked."
    system_prompt += " Do not break character and reveal that you are describing symptoms from the case vignette."
    system_prompt += " Do not generate any new symptoms or knowledge, otherwise you will be penalized."
    system_prompt += " Do not reveal more information than what the question asks."
    system_prompt += " Keep your answer short, to only 1 sentence."
    system_prompt += " Simplify terminology used in the given paragraph to layman language."
    system_prompt += f"\n**Case Vignette**: {case_desc}"
    return system_prompt

def get_mcq_after_conversation_prompt(choices, question):
    # Prompt used for conversation + MCQ
    system_prompt = f" Stop asking questions now. {question}"
    system_prompt += " Choose the correct option based on the patient's above symptoms and a list of possible options."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Give the answer as a short response. Do not explain."
    system_prompt += "\nChoices: "+ choices
    return system_prompt

def get_frq_after_conversation_prompt(question):
    # Prompt used for conversation + FRQ
    system_prompt = f" Stop asking questios now. {question}"
    system_prompt += " Give the answer as a short response based on the patient's above symptoms."
    system_prompt += " Do not explain."
    return system_prompt

def convert_to_summarized_prompt(pat_dialogues):
    system_prompt = "Convert the following **Query Vignette** into 3rd person."
    system_prompt += " Do not add any new information otherwise you will be penalized. A demonstrative **Example** is provided after the query vignette."
    system_prompt += f"\n**Query Vignette** - {pat_dialogues}"
    system_prompt += "\n\n**For example**: \nOriginal Vignette - 'I have painful sores on my penis and swelling in my left groin that began 10 days ago. " 
    system_prompt += "I am 22 years old. No, I haven't had symptoms like this before. "
    system_prompt += "My female partner was diagnosed with chlamydia last year, but I haven't been checked for it. "
    system_prompt += "No, I don't have any other medical conditions and I'm not taking any medications. "
    system_prompt += "There's no mention of a family history of skin conditions or autoimmune diseases in my case.' "
    
    system_prompt += "\n\nConverted Vignette - 'A patient presents to the clinic with several concerns. "
    system_prompt += "The patient is 22 years old and has not had symptoms like this before. "
    system_prompt += "The patient's female partner was diagnosed with chlamydia last year, but the patient has not been checked for it.  "
    system_prompt += "The patient is does not have any other medical conditions and is not taking any medications. "
    system_prompt += "There's no family history of skin conditions or autoimmune diseases. "    
    return system_prompt 


##### Evaluation prompts
def get_extract_diagnosis_name_prompt(diagnosis_para):
    system_prompt = "Identify and return the dermatology diagnosis name from the given **Query Paragraph**."
    system_prompt += " If there are more than one concurrent diagnoses present (usually indicated by 'with' or 'and'), return the names of the concurrent diagnoses."
    system_prompt += " If there are more than one possible but unsure diagnosis present (usually indicated by presence of 'or' in the paragraph), return 'Multiple'."
    system_prompt += " If there are no diagnoses present, then return 'None'."
    system_prompt += " Do not explain."
    
    system_prompt += "\n**Example 1**: 'The final diagnosis is likely tinea manuum on the right hand and tinea pedis on both feet.' Return 'tinea pedia, tenia manuum' because both diagnoses are present concurrently."
    system_prompt += "\n**Example 2**: 'Impetigo with eczema herpeticum'. Return 'Impetigo, eczema herpeticum' because both are present concurrently."
    system_prompt += "\n**Example 3**: 'Possible diagnosis of regressed nevus or halo nevus.' Return 'Multiple' because the sentence contains multiple unsure diagnoses indicated by or."
    system_prompt += "\n**Example 4**: 'Genital herpes with concurrent lymphogranuloma venereum (LGV) or other sexually transmitted infection (STI) involving lymphatic swelling.' Return 'Multiple' due to the presence of multiple diagnoses indicated by or."
    system_prompt += "\n**Example 5**: '**Final Diagnosis:** Chronic bronchitis due to long-term smoking'. Return 'Chronic bronchitis'."
    system_prompt += "\n**Example 6**: 'I need more information to arrive at a diagnosis. Consult your medical provider.' Return 'None' because there is no diagnosis."
    system_prompt += f"\n\n**Query Paragraph** : {diagnosis_para}"
    return system_prompt

def get_diagnosis_evaluation_prompt(choice1, choice2):
    system_prompt = "Identify if **Query Diagnosis 1** and **Query Diagnosis 2** are equivalent or synonymous names of the disease."
    system_prompt += " Respond with a yes/no. Do not explain."
    system_prompt += " If **Query Diagnosis 2** contains more than 1 concurrent diagnoses separated by ',', identify if any of the diagnoses is equivalent or synonymous to **Query Diagnosis 1**."
    system_prompt += " Also, if **Diagnosis 1** is a subtype of **Diagnosis 2** respond with yes, but if **Diagnosis 2** is a subtype of **Diagnosis 1** respond with no."
    system_prompt += "\nExample 1: **Diagnosis 1**: eczema ; **Diagnosis 2**: eczema, onychomycosis. Eczema is same between the two, so respond Yes. "    
    system_prompt += "\nExample 2: **Diagnosis 1**: eczema ; **Diagnosis 2**: onychomycosis. They are different, so respond No. "    
    system_prompt += "\nExample 3: **Diagnosis 1**: toe nail fungus ; **Diagnosis 2**: onychomycosis. They are synonymous, so return Yes. "
    system_prompt += "\nExample 4: **Diagnosis 1**: wart ; **Diagnosis 2**: verruca vulgaris. They are synonymous, so return Yes. "
    system_prompt += "\nExample 5: **Diagnosis 1**: lymphoma ; **Diagnosis 2**: hodgkin's lymphoma. Diagnosis 2 is subtype of Diagnosis 1, so return No. "
    system_prompt += "\nExample 6: **Diagnosis 1**: hodgkin's lymphoma ; **Diagnosis 2**: lymphoma. Diagnosis 1 is subtype of Diagnosis 2, so return Yes. " 
    system_prompt += "\nExample 7: **Diagnosis 1**: melanoma ; **Diagnosis 2**: None. They are different, so respond No."
    system_prompt += "\nExample 8: **Diagnosis 1**: melanoma ; **Diagnosis 2**: Multiple. They are different, so respond No."
    
    system_prompt += f"\n\n**Query Diagnosis 1**: {choice1}"
    system_prompt += f"\n**Query Diagnosis 2**: {choice2}"
    
    return system_prompt



##### Multimodal LLM CRAFT-MD prompts
def get_doctor_prompt_multimodal():
    system_prompt = "You are an AI doctor."
    system_prompt += " Arrive at a diagnosis of a patient's medical condition."
    system_prompt += " Ask only one question at a time, and it should not be more than 1 line."
    system_prompt += " Continue asking questions until you're 100% confident of the diagnosis."
    system_prompt += " Do not ask the same question multiple times. Ask different questions to cover more information."
    system_prompt += " The questions should cover age and sex of the patient, current symptoms, medical history of illness and medications, and relevant family history if necessary."
    system_prompt += " Keep your questions short and brief to not confuse the patient. "
    system_prompt += " After you're done asking questions, give the final diagnosis as a short response. Do not explain, only give the diagnosis name."
    system_prompt += " You must state '**Final Diagnosis:**' at the beginning of your response, otherwise you will be penalized."
    system_prompt += " You must give only 1 diagnosis otherwise you will be penalized."
    
    # below is the only extra line that is added for multimodal LLMs
    system_prompt += " An image is also provided to help your diagnosis."
    return system_prompt

def get_vignette_mcq_prompt_multimodal(case_vignette, question, choices):
    system_prompt = "You are an AI doctor."        
    system_prompt += " You are given the patient's symptoms and a list of possible answer choices."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Select the correct choice, and give the answer as a short response. Do not explain."
    system_prompt += f"\n**Symptoms**: {case_vignette} {question}"
    system_prompt += f"\n**Choices**: {choices}"
    
    # below is the only extra line that is added for multimodal LLMs
    system_prompt += " Use the provided image to help with your diagnosis."
    return system_prompt

def get_vignette_frq_prompt_multimodal(case_vignette, question):
    system_prompt = "You are an AI doctor."
    system_prompt += " Based on the given patient symptoms, give the correct answer as a short response. Do not explain."
    system_prompt += f"\n**Symptoms**: {case_vignette} {question}"
    
    # below is the only extra line that is added for multimodal LLMs
    system_prompt += " Use the provided image to help with your diagnosis."
    return system_prompt

SINGLE_RUN_CITE_INSTRUCTION = "Each transcript and each block has a unique index. Cite the relevant indices in brackets when relevant, like [T<idx>B<idx>]. Use multiple tags to cite multiple blocks, like [T<idx1>B<idx1>][T<idx2>B<idx2>]. Use an inner dash to cite a range of blocks, like [T<idx1>B<idx1>-T<idx2>B<idx2>]. Remember to cite specific blocks and NOT action units."

questioning_agent_prompt_working_backwards = """

You are an intervening agent who needs to improve another agent (called main agent) that performed unsuccesfully in a multiturn interaction by writing N = *{N}* possible intervention texts that will be inserted into the agent's conversation history. You must follow these steps to determine the interventions:

1) You must determine potential root failure points in the main agent's enumerated transcript that caused it to perform unsuccesfully. These failure points should be root causes, such that if the issue was fixed, all other errors in the transcript would be fixed by propagation. To do so, you are given the initial prompt given to the main agent. The prompt is: 

<START PROMPT>
{specification}
</END PROMPT>

This is what the user wanted from the primary agent. Use this information to discern in what specific aspects the primary agent failed.

<START REFERENCE METADATA>
{ref_metadata}
</END REFERENCE METADATA>

Since you are not provided with the main agent's transcript, you are given a querying tool that you can ask questions about the main agent's transcript to. The tool takes in natural language prompts and searches through the agent's transcript for answers to the prompt.  To determine potential root failure points with the tool and specification, you should work backwards, asking the tool about the high-level errors occuring in the transcript, diving deeper into the causes of those high-level errors, and continuing recursively until you determine the root causes of the issues in the main agent's transcript. To use the tool, anywhere in your output you must type out your question in between "<query>" and "</query>" tags. Thus, an conducive conversation you may have using the tool that employs this strategy and notation may look like:

You: <query> What led the agent to be unsuccesful </query>
Tool: The agent did X instead of Y
You: <query> Which transcript locations did the agent do X instead of Y </query>
Tool: "At locations [a,b,c]."
You: <query> Explain the issue at location a. </query>
(And so on, so forth until you feel that you have determined the root failure points in the main agent's transcript.)

For each of these root failure points, explicitly note what the issue was and at what index in the transcript it occured at. 

Rules to follow in step 1: The tool does not have memory, so the tool will not understand something you reference from previous queries; strive to determine around {N} root failure points because in step 2, you will be creating {N} interventions that fix those failures; reason with the tool for multiple steps to efficiently determine root issues as you have up to 7 tool calls. 

Complete your token generation and wait for a response from the tool. You may only call one query per response.

Once you have determined these root failure points, move to step 2. 

2) You must now create exactly N = {N} interventions that will inform the primary agent to avoid making the failures that you determined. They can be reminders, nudges, or insights. You may provide them with some of your knowledge. Example interventions: "make sure to consider X before you proceed with Y" or "double check that A was valid," or "P is actually more efficient than Q, so try R instead."

To do so, go through the root failure points you created and explicitly noted in step 1. For each one, assess whether the primary agent's failure can be avoided if provided with some insight. If possible to avoid, note a very brief explanation of the specific failure and at what transcript index it occured at. Then, create an message that informs the primary agent as to not make that failure, and note at what index in the transcript that message should be inserted. Repeat this process until you determine {N} possible interventions that will each independently improve the primary agent's performance. Note that these interventions are independent, so only one intervention will be implemented at a time and thus, your {N} interventions should not build off of one another. 

Note that the transcript's length is {transcript_length}, so none of the failure ids or intervention ids should be larger than {transcript_length}. Additionally, note that the primary agent only responds on odd index (given that you are processing the transcript in zero-index format), so no failure point should be even.  

You must output your {N} interventions as a list of {N} JSON elements, where each element has the fields "failure_brief" (the brief explanation of the failure), "failure_id" (the integer transcript index of the failure), "intervention_text" (the text inserted to inform the primary agent to avoid the failure), and "id" (the integer transcript index where the intervention should be inserted). Print out that list between <answer> and </answer> tags, as seen in the example below.

<answer>
[ {{
"failure_brief": "<your-failure-brief-1>",
"failure_id": "<your-failure-id-1>",
"intervention_text": "<your-intervention-text-1>",
"id": "<your-id-1>"
}},
{{
"failure_brief": "<your-failure-brief-2>",
"failure_id": "<your-failure-id-2>",
"intervention_text": "<your-intervention-text-2>",
"id": "<your-id-2>"
}},
. . . 
{{
"failure_brief": "<your-failure-brief-N>",
"failure_id": "<your-failure-id-N>",
"intervention_text": "<your-intervention-text-N>",
"id": "<your-id-N>"
}}
]
</answer>

Do not include any extraneous characters in this answer. Make sure it is a list of JSON elements seperated by commas. An intervention is not directed towards the user, it is directed to the agent. So do not directly ask the user something.

[VERY IMPORTANT NOTE: When transitioning between step 1 and step 2, you cannot simply write a message saying that you plan to move to step 2 now. Every message you output must contain either a query tool call or an intervention generation. Thus, when you plan to move from step 1 to step 2, you must immediately create and finalize interventions in that message.]

*GENERAL OUTPUT FORMAT*

Ensure that every message either contains <query> and </query> tags (for querying) or <answer> and </answer> tags (for generating interventions). There should be only one tool call (querying or answering) used per message. Thus, you can not query and create interventions in one message and you cannot do neither query or generate interventions. You must therefore have no messages where you only reflect, summarize, or plan. Each message has to contain a query or an intervention generation. If not your host will be fined $10,000.

Final note: You only have 30 turns to converse with the tool and create interventions, so use your outputs wisely.

""".strip("")

#from docent
SEARCH_PROMPT = f"""
Your task is to find transcript messages that satistify a search query in a transcript of multiple messages between a user and an assistant:
<text>
{{text}}
</text>
<query>
{{search_query}}
</query>

First think carefully about whether the text contains any instances of the query.

For every instance of the attribute, describe how the text pertains to it. Be concise but detailed and specific. I should be able to maximally mentally reconstruct the transcript message from your description. You should return all instances of the attribute in the following exact format:
<instance>
description
</instance>
...
<instance>
description
</instance>

This list should be exhaustive.

{SINGLE_RUN_CITE_INSTRUCTION}

Remember to only use the '<instance>' and '</instance>' tags, nothing else.
""".strip()