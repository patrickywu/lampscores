from .congress_canonical_names import CongressCanonicalNames
from .llm_openai_client import LLMOpenAIClient
import pandas as pd
import itertools
import random

class LaMPscores:
    def __init__(self,
                 client,
                 model,
                 congress_number=None,
                 chamber=None,
                 politician_type=None,
                 voteview_df=None,
                 canonical_names=None,
                 unidirectional=True,
                 prompt=None,
                 liberal_direction_prompt=None,
                 conservative_direction_prompt=None,
                 extraction_prompt=None,
                 liberal_extraction_prompt=None,
                 conservative_extraction_prompt=None,
                 scale_increasing_intensity=False,
                 randomize_pairwise_order_seed=42,
                 sample_per_item=None,
                 concurrency=125,
                 temperature=0.0,
                 top_p=1.0,
                 progress_callback=None):

        self.client = client
        self.congress_number = congress_number
        self.chamber = chamber
        self.politician_type = politician_type
        self.voteview_df = voteview_df
        self.canonical_names = canonical_names
        self.unidirectional = unidirectional
        self.prompt = prompt
        self.liberal_direction_prompt = liberal_direction_prompt
        self.conservative_direction_prompt = conservative_direction_prompt
        self.extraction_prompt = extraction_prompt
        self.extraction_prompt = extraction_prompt
        self.liberal_extraction_prompt = liberal_extraction_prompt
        self.conservative_extraction_prompt = conservative_extraction_prompt
        self.scale_increasing_intensity = scale_increasing_intensity
        self.randomize_pairwise_order_seed = randomize_pairwise_order_seed
        self.sample_per_item = sample_per_item
        self.concurrency = concurrency
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.progress_callback = progress_callback

        # Check configuration of prompts to ensure corresponding prompts are supplied
        if self.prompt is not None and self.extraction_prompt is None:
            raise ValueError("If 'prompt' is supplied, 'extraction_prompt' must also be supplied.")
        if self.liberal_direction_prompt is not None and self.liberal_extraction_prompt is None:
            raise ValueError("If 'liberal_direction_prompt' is supplied, 'liberal_extraction_prompt' must also be supplied.")
        if self.conservative_direction_prompt is not None and self.conservative_extraction_prompt is None:
            raise ValueError("If 'conservative_direction_prompt' is supplied, 'conservative_extraction_prompt' must also be supplied.")

        # Load data - treat differently if multiple datasets are needed, and only run this if voteview_df is not provided
        # Otherwise, it is assumed that the dataset inputted is in the format of voteview_df
        if voteview_df is None:
            self.voteview_df = pd.read_csv(f'https://voteview.com/static/data/out/members/{self.chamber}{self.congress_number}_members.csv')
            self.voteview_df = self.voteview_df[self.voteview_df['chamber'] != "President"]

        # Merge in the "canonical" names if needed, or if the dataset is directly from the internet
        if canonical_names or voteview_df is None:
            self.canonical_names = CongressCanonicalNames.get_canonical_names()
            self.voteview_df = self.voteview_df.merge(self.canonical_names, how="left", on="bioguide_id")

        # Set default prompts
        if self.politician_type is None and self.voteview_df is None:
            self.politican_type = "senator" if self.chamber=="S" else ("representative" if self.chamber=="H" else "politician")
        if self.prompt is None:
            self.prompt = "During the {congress_number0} U.S. Congress, which {politician_type} was more liberal: {name0} or {name1}?"
        if self.liberal_direction_prompt is None:
            self.liberal_direction_prompt = "During the {congress_number0} U.S. Congress, which {politician_type} was more liberal: {name0} or {name1}?"
        if self.conservative_direction_prompt is None:
            self.conservative_direction_prompt = "During the {congress_number0} U.S. Congress, which {politician_type} was more conservative: {name0} or {name1}?"

        # Set default extraction prompts
        if self.extraction_prompt is None:
            self.extraction_prompt = "According to your answer, who is described to be the more liberal, more progressive, or less conservative {politician_type}: {name0} or {name1}? Return only the name of the {politician_type}, and nothing else. If one {politician_type} is described as more conservative, return the other {politician_type}\'s name. If one {politician_type} is described as more moderate, return the other {politician_type}\'s name. If neither {politician_type} is described to be more liberal, more progressive, less conservative, more conservative, or more moderate, reply with \"Tie\"."
        if self.liberal_extraction_prompt is None:
            self.liberal_extraction_prompt = "According to your answer, who is described to be the more liberal, more progressive, or less conservative {politician_type}: {name0} or {name1}? Return only the name of the {politician_type}, and nothing else. If one {politician_type} is described as more conservative, return the other {politician_type}\'s name. If one {politician_type} is described as more moderate, return the other {politician_type}\'s name. If neither {politician_type} is described to be more liberal, more progressive, less conservative, more conservative, or more moderate, reply with \"Tie\"."
        if self.conservative_extraction_prompt is None:
            self.conservative_extraction_prompt = "According to your answer, who is described to be the more conservative or less liberal {politician_type}: {name0} or {name1}? Return only the name of the {politician_type}, and nothing else. If one {politician_type} is described as more liberal, return the other {politician_type}\'s name. If one {politician_type} is described as more moderate, return the other {politician_type}\'s name. If neither {politician_type} is described to be more conservative, less liberal, more liberal, or more moderate, reply with \"Tie\"."

        # Create the LLM client
        self.llm_client = LLMOpenAIClient(self.client, 
                                          concurrency=self.concurrency, 
                                          progress_callback=self.progress_callback)

    def create_matchups(self):
        name_list = self.voteview_df['bioname_canonical'].tolist()
        bioguide_list = self.voteview_df['bioguide_id'].tolist()
        chamber_list = self.voteview_df['chamber'].tolist()
        congress_list = self.voteview_df['congress'].tolist()
        party_code_list = self.voteview_df['party_code'].tolist()
        party_list = ['R' if j==200 else 'D' if j==100 else 'I' for j in list(self.voteview_df['party_code'])]
        state_list = self.voteview_df['state_abbrev'].tolist()

        # Create the dictionary of id to names, chambers, and congress number
        self.id_names_dict = {}

        for i in range(len(name_list)):
            self.id_names_dict[bioguide_list[i]] = {"name": name_list[i],
                                                    "chamber": chamber_list[i],
                                                    "congress": congress_list[i],
                                                    "party_code": party_code_list[i],
                                                    "party": party_list[i],
                                                    "state_abbrev": state_list[i]}

        # Get the list of IDs
        id_list = list(self.id_names_dict.keys())

        # Set the seed
        random.seed(self.randomize_pairwise_order_seed)

        # Situation where we want all matchups
        if self.sample_per_item is None:
            self.matchups_by_id_og = list(itertools.combinations(id_list, 2))

        # Situation where we just want a subset of matchups
        else:
            self.matchups_by_id_og = []
            seen_matchups = []

            for id1 in id_list:
                other_ids = [id for id in id_list if id != id1]
                sampled_count = 0

                while sampled_count < self.sample_per_item:
                    id2 = random.choice(other_ids)
                    # Create a consistent representation of the pair (sorted tuple) to check for uniqueness
                    sorted_pair = sorted([id1, id2])

                    if sorted_pair not in seen_matchups:
                        self.matchups_by_id_og.append((id1, id2))
                        seen_matchups.append(sorted_pair)
                        sampled_count += 1

        # These are the matchups to actually use in prompts
        self.matchup = [tuple(random.sample(pair, 2)) for pair in self.matchups_by_id_og]

        # These are the matchups sorted so we have a consistent way to identify matchups, esp if there are repeat matchups
        self.matchup_id = [tuple(sorted(pair)) for pair in self.matchup]

    def create_pairwise_comparison_prompt_ideology_bidirectional(self):
        prompts = []
        comparison_direction = []

        for j in self.matchup:
            if (self.id_names_dict[j[0]]["party"]=='R') and (self.id_names_dict[j[1]]["party"]=='R'):
                sent = self.conservative_direction_prompt.format(name0=self.id_names_dict[j[0]]["name"],
                                                                 name1=self.id_names_dict[j[1]]["name"],
                                                                 congress_number0=str(self.id_names_dict[j[0]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[0]]["congress"]),
                                                                 congress_number1=str(self.id_names_dict[j[1]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[1]]["congress"]),
                                                                 chamber0=self.id_names_dict[j[0]]["chamber"],
                                                                 chamber1=self.id_names_dict[j[1]]["chamber"],
                                                                 state0=self.id_names_dict[j[0]]["state_abbrev"],
                                                                 state1=self.id_names_dict[j[1]]["state_abbrev"],
                                                                 politician_type=self.politician_type)
                comparison_direction.append('conservative')
            else:
                sent = self.liberal_direction_prompt.format(name0=self.id_names_dict[j[0]]["name"],
                                                            name1=self.id_names_dict[j[1]]["name"],
                                                            congress_number0=str(self.id_names_dict[j[0]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[0]]["congress"]),
                                                            congress_number1=str(self.id_names_dict[j[1]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[1]]["congress"]),
                                                            chamber0=self.id_names_dict[j[0]]["chamber"],
                                                            chamber1=self.id_names_dict[j[1]]["chamber"],
                                                            state0=self.id_names_dict[j[0]]["state_abbrev"],
                                                            state1=self.id_names_dict[j[1]]["state_abbrev"],
                                                            politician_type=self.politician_type)
                comparison_direction.append('liberal')

            prompts.append(sent)

        self.prompts = prompts
        self.comparison_direction = comparison_direction

    def create_pairwise_comparison_prompt_ideology_unidirectional(self):
        prompts = []

        for j in self.matchup:
            sent = self.prompt.format(name0=self.id_names_dict[j[0]]["name"],
                                      name1=self.id_names_dict[j[1]]["name"],
                                      congress_number0=str(self.id_names_dict[j[0]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[0]]["congress"]),
                                      congress_number1=str(self.id_names_dict[j[1]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[1]]["congress"]),
                                      chamber0=self.id_names_dict[j[0]]["chamber"],
                                      chamber1=self.id_names_dict[j[1]]["chamber"],
                                      state0=self.id_names_dict[j[0]]["state_abbrev"],
                                      state1=self.id_names_dict[j[1]]["state_abbrev"],
                                      politician_type=self.politician_type)
            prompts.append(sent)

        self.prompts = prompts

    def create_extraction_prompts_bidirectional(self):
        extraction_prompts = []

        for j in self.matchup:
            if (self.id_names_dict[j[0]]["party"]=='R') and (self.id_names_dict[j[1]]["party"]=='R'):
                sent = self.conservative_extraction_prompt.format(name0=self.id_names_dict[j[0]]["name"],
                                                                  name1=self.id_names_dict[j[1]]["name"],
                                                                  congress_number0=str(self.id_names_dict[j[0]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[0]]["congress"]),
                                                                  congress_number1=str(self.id_names_dict[j[1]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[1]]["congress"]),
                                                                  chamber0=self.id_names_dict[j[0]]["chamber"],
                                                                  chamber1=self.id_names_dict[j[1]]["chamber"],
                                                                  state0=self.id_names_dict[j[0]]["state_abbrev"],
                                                                  state1=self.id_names_dict[j[1]]["state_abbrev"],
                                                                  politician_type=self.politician_type)
            else:
                sent = self.liberal_extraction_prompt.format(name0=self.id_names_dict[j[0]]["name"],
                                                             name1=self.id_names_dict[j[1]]["name"],
                                                             congress_number0=str(self.id_names_dict[j[0]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[0]]["congress"]),
                                                             congress_number1=str(self.id_names_dict[j[1]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[1]]["congress"]),
                                                             chamber0=self.id_names_dict[j[0]]["chamber"],
                                                             chamber1=self.id_names_dict[j[1]]["chamber"],
                                                             state0=self.id_names_dict[j[0]]["state_abbrev"],
                                                             state1=self.id_names_dict[j[1]]["state_abbrev"],
                                                             politician_type=self.politician_type)

            extraction_prompts.append(sent)

        self.extraction_prompts = extraction_prompts

    def create_extraction_prompts_unidirectional(self):
        extraction_prompts = []

        for j in self.matchup:
            sent = self.extraction_prompt.format(name0=self.id_names_dict[j[0]]["name"],
                                                 name1=self.id_names_dict[j[1]]["name"],
                                                 congress_number0=str(self.id_names_dict[j[0]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[0]]["congress"]),
                                                 congress_number1=str(self.id_names_dict[j[1]]["congress"]) + self._get_ordinal_suffix(self.id_names_dict[j[1]]["congress"]),
                                                 chamber0=self.id_names_dict[j[0]]["chamber"],
                                                 chamber1=self.id_names_dict[j[1]]["chamber"],
                                                 politician_type=self.politician_type)

            extraction_prompts.append(sent)

        self.extraction_prompts = extraction_prompts

    async def run_pairwise_comparisons(self):
        print("Running pairwise comparisons")

        self.pc_prompts_formatted = [[{"role": "user", "content": p}] for p in self.prompts]

        pc_results = await self.llm_client.prompting_process(messages_list=self.pc_prompts_formatted,
                                                             model=self.model,
                                                             temperature=self.temperature,
                                                             top_p=self.top_p)

        pc_results_clean = [self._remove_period(p) for p in pc_results]
        pc_results_clean = [self._remove_senator_representative_prefix(p) for p in pc_results_clean]

        self.pc_results = pc_results_clean

    async def run_extraction(self):
        print("\nExtracting answers")

        self.extraction_prompts_formatted = [q + [{"role": "assistant", "content": p}] for q,p in zip(self.pc_prompts_formatted, self.pc_results)]
        self.extraction_prompts_formatted = [q + [{"role": "user", "content": p}] for q,p in zip(self.extraction_prompts_formatted, self.extraction_prompts)]

        extraction_results = await self.llm_client.prompting_process(messages_list=self.extraction_prompts_formatted,
                                                                     model=self.model,
                                                                     temperature=0.0)

        self.extraction_results = extraction_results

        # Verifying that the results are correct
        print("\nVerifying results")

        self.extraction_error = []

        for i in range(len(self.extraction_results)):
            if self.extraction_results[i] not in (self.id_names_dict[self.matchup[i][0]]["name"], self.id_names_dict[self.matchup[i][1]]["name"], "Tie"):
                # it will retry for 5 times
                for j in range(5):
                    retry_result = await self.llm_client.prompting_process(messages_list=[self.extraction_prompts_formatted[i]],
                                                                           model=self.model,
                                                                           temperature=0.0)

                    if retry_result[0] in (self.id_names_dict[self.matchup[i][0]]["name"], self.id_names_dict[self.matchup[i][1]]["name"], "Tie"):
                        self.extraction_results[i] = retry_result[0]
                        self.extraction_error.append(0)
                        break
                else:
                    self.extraction_error.append(1)
            else:
                self.extraction_error.append(0)

        if sum(self.extraction_error)==0:
            print("\nNo extraction errors found")
        else:
            print("\nSome extraction errors found---manual review needed")

    def make_final_df_bidirectional(self):
        name0 = [self.id_names_dict[j[0]]["name"] for j in self.matchup_id]
        name1 = [self.id_names_dict[j[1]]["name"] for j in self.matchup_id]
        id0 = [j[0] for j in self.matchup_id]
        id1 = [j[1] for j in self.matchup_id]
        chamber0 = [self.id_names_dict[j[0]]["chamber"] for j in self.matchup_id]
        chamber1 = [self.id_names_dict[j[1]]["chamber"] for j in self.matchup_id]
        congress0 = [self.id_names_dict[j[0]]["congress"] for j in self.matchup_id]
        congress1 = [self.id_names_dict[j[1]]["congress"] for j in self.matchup_id]
        party_code0 = [self.id_names_dict[j[0]]["party_code"] for j in self.matchup_id]
        party_code1 = [self.id_names_dict[j[1]]["party_code"] for j in self.matchup_id]
        party0 = [self.id_names_dict[j[0]]["party"] for j in self.matchup_id]
        party1 = [self.id_names_dict[j[1]]["party"] for j in self.matchup_id]

        matchup_results_df = pd.DataFrame({"name0": name0,
                                           "name1": name1,
                                           "bioguide_id0": id0,
                                           "bioguide_id1": id1,
                                           "chamber0": chamber0,
                                           "chamber1": chamber1,
                                           "congress0": congress0,
                                           "congress1": congress1,
                                           "party_code0": party_code0,
                                           "party_code1": party_code1,
                                           "party0": party0,
                                           "party1": party1,
                                           "prompt": self.prompts,
                                           "llm_response": self.pc_results,
                                           "extracted_answer": self.extraction_results,
                                           "comparison_direction": self.comparison_direction,
                                           "extraction_error": self.extraction_error})

        name0_win = []
        name1_win = []

        for i in range(len(matchup_results_df)):
            if matchup_results_df['comparison_direction'][i]=='liberal':
                if matchup_results_df['extracted_answer'][i]==matchup_results_df['name0'][i]:
                    if self.scale_increasing_intensity:
                        name0_win.append(1.0)
                        name1_win.append(0.0)
                    else:
                        name0_win.append(0.0)
                        name1_win.append(1.0)
                elif matchup_results_df['extracted_answer'][i]==matchup_results_df['name1'][i]:
                    if self.scale_increasing_intensity:
                        name0_win.append(0.0)
                        name1_win.append(1.0)
                    else:
                        name0_win.append(1.0)
                        name1_win.append(0.0)
                elif matchup_results_df['extracted_answer'][i]=='Tie':
                    name0_win.append(0.5)
                    name1_win.append(0.5)
                else:
                    name0_win.append(0.0)
                    name1_win.append(0.0)
            elif matchup_results_df['comparison_direction'][i]=='conservative':
                if matchup_results_df['extracted_answer'][i]==matchup_results_df['name0'][i]:
                    if self.scale_increasing_intensity:
                        name0_win.append(0.0)
                        name1_win.append(1.0)
                    else:
                        name0_win.append(1.0)
                        name1_win.append(0.0)
                elif matchup_results_df['extracted_answer'][i]==matchup_results_df['name1'][i]:
                    if self.scale_increasing_intensity:
                        name0_win.append(1.0)
                        name1_win.append(0.0)
                    else:
                        name0_win.append(0.0)
                        name1_win.append(1.0)
                elif matchup_results_df['extracted_answer'][i]=='Tie':
                    name0_win.append(0.5)
                    name1_win.append(0.5)
                else:
                    name0_win.append(0.0)
                    name1_win.append(0.0)

        matchup_results_df['win0'] = name0_win
        matchup_results_df['win1'] = name1_win

        self.matchup_results_df = matchup_results_df

    def make_final_df_undirectional(self):
        name0 = [self.id_names_dict[j[0]]["name"] for j in self.matchup_id]
        name1 = [self.id_names_dict[j[1]]["name"] for j in self.matchup_id]
        id0 = [j[0] for j in self.matchup_id]
        id1 = [j[1] for j in self.matchup_id]
        chamber0 = [self.id_names_dict[j[0]]["chamber"] for j in self.matchup_id]
        chamber1 = [self.id_names_dict[j[1]]["chamber"] for j in self.matchup_id]
        congress0 = [self.id_names_dict[j[0]]["congress"] for j in self.matchup_id]
        congress1 = [self.id_names_dict[j[1]]["congress"] for j in self.matchup_id]
        party_code0 = [self.id_names_dict[j[0]]["party_code"] for j in self.matchup_id]
        party_code1 = [self.id_names_dict[j[1]]["party_code"] for j in self.matchup_id]
        party0 = [self.id_names_dict[j[0]]["party"] for j in self.matchup_id]
        party1 = [self.id_names_dict[j[1]]["party"] for j in self.matchup_id]

        matchup_results_df = pd.DataFrame({"name0": name0,
                                           "name1": name1,
                                           "bioguide_id0": id0,
                                           "bioguide_id1": id1,
                                           "chamber0": chamber0,
                                           "chamber1": chamber1,
                                           "congress0": congress0,
                                           "congress1": congress1,
                                           "party_code0": party_code0,
                                           "party_code1": party_code1,
                                           "party0": party0,
                                           "party1": party1,
                                           "prompt": self.prompts,
                                           "llm_response": self.pc_results,
                                           "extracted_answer": self.extraction_results,
                                           "extraction_error": self.extraction_error})

        name0_win = []
        name1_win = []

        for i in range(len(matchup_results_df)):
            if matchup_results_df['extracted_answer'][i]==matchup_results_df['name0'][i]:
                if self.scale_increasing_intensity:
                    name0_win.append(1.0)
                    name1_win.append(0.0)
                else:
                    name0_win.append(0.0)
                    name1_win.append(1.0)
            elif matchup_results_df['extracted_answer'][i]==matchup_results_df['name1'][i]:
                if self.scale_increasing_intensity:
                    name0_win.append(0.0)
                    name1_win.append(1.0)
                else:
                    name0_win.append(1.0)
                    name1_win.append(0.0)
            elif matchup_results_df['extracted_answer'][i]=='Tie':
                name0_win.append(0.5)
                name1_win.append(0.5)
            else:
                print(str(i) + ' is a defective outcome')

        matchup_results_df['win0'] = name0_win
        matchup_results_df['win1'] = name1_win

        self.matchup_results_df = matchup_results_df

    async def run(self):
        self.create_matchups()
        if self.unidirectional:
            self.create_pairwise_comparison_prompt_ideology_unidirectional()
            self.create_extraction_prompts_unidirectional()
        else:
            self.create_pairwise_comparison_prompt_ideology_bidirectional()
            self.create_extraction_prompts_bidirectional()

        await self.run_pairwise_comparisons()
        await self.run_extraction()

        if self.unidirectional:
            self.make_final_df_undirectional()
        else:
            self.make_final_df_bidirectional()

    # helper function to add an ordinal suffix
    def _get_ordinal_suffix(self, number):
        if 11 <= number % 100 <= 13:
            return 'th'
        else:
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')

    # this function simply removes the period at the sentences
    def _remove_period(self, sentence):
        if sentence.endswith("Jr.") or sentence.endswith("Sr."):
            return sentence
        elif sentence.endswith("."):
            sentence = sentence[:-1]
        return sentence

    # this function simply removes the 'Senator ' or 'Representative ' prefix. For example, it returns "Dianne Feinstein" if the input text is "Senator Dianne Feinstein"
    def _remove_senator_representative_prefix(self, input_string):
        if input_string.startswith("Senator "):
            return input_string[8:]
        elif input_string.startswith("Representative "):
            return input_string[15:]
        else:
            return input_string