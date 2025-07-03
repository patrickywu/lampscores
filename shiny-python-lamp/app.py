import io
import asyncio
from openai import AsyncOpenAI
import traceback
import sys
from contextlib import contextmanager

from shiny import App, ui, reactive, render

from lampscores import LaMPscores

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
congress_choices = {str(n): str(n) for n in range(37, 120)}

## Model Configuration Card
model_config_card = ui.card(
    ui.card_header("Provider, API Key, and Model"),
    # Selecting the API Provider
    ui.input_select("provider", "API Provider", {"openai": "OpenAI", "deepinfra": "DeepInfra"}),
    # Inputting API key
    ui.input_password('api_key', "API Key for Provider", placeholder="sk-..."),
    # Model
    ui.input_text("model", "Model", placeholder="gpt-4.1-mini"),
    ui.input_slider("temperature", "Temperature", 0.0, 2.0, 1.0, step=0.1),
    ui.input_slider("top_p", "top‑p", 0.0, 1.0, 1.0, step=0.05),
    ui.input_slider("concurrency", "Number of calls in parallel", 1, 200, 125, step=1)
)

## Voteview Configuration Card
voteview_config_card = ui.card(
    ui.card_header("Data"),
    ui.input_checkbox('use_own_voteview', 'Upload my own Voteview data instead'),
    ui.panel_conditional(
        "!input.use_own_voteview",
        # Congress number
        ui.input_select("congress_numbers", "Congress Number", congress_choices, multiple=False, selected="116"),
        # Chamber
        ui.input_select("chamber", "Chamber", {"S": "Senate", "H": "House", "HS": "Both"})
    ),
    ui.panel_conditional(
        "input.use_own_voteview",
        ui.tooltip(ui.input_file("custom_voteview_data", 'Upload Voteview CSV', accept=['.csv']), "CSV must match Voteview's format. For more information, see https://voteview.com/data."),
        ui.tooltip(ui.input_text("custom_politician_type", "What to call politicians in pairwise comparisons", placeholder="politician"), "For example, \"senator\" or \"representative\""),
        ui.tooltip(ui.input_checkbox("use_canonical_names", "Merge in canonical (i.e., commonly used) names"), "Canonical names may be provided as a variable \"bioname_canonical\". Canonical names are the names we commonly refer to politicians as, such as the title of their Wikipedia page. If canonical names are not provided, check to automatically merge these in.")
    ),
    ui.input_text("num_sample", "Number of samples per politician. Leave blank to make all pairwise comparisons."),
    ui.tooltip(ui.input_text("seed", "Random seed for sampling and pairing", value="42", placeholder="42"), "Defaults to seed 42")
)

## Prompt Configuration Card
prompt_config_card = ui.card(
    ui.card_header("Pairwise Comparison Prompt Configuration"),
    ui.tooltip(ui.input_checkbox("bidirectional_comparisons", "Use bidirectional pairwise comparisons", width="400px"), "Checking this option allows you to have different prompts when comparing Democrats against Democrats or Republicans, and Republicans against Republicans. For example, some LLMs will fail to compare two conservatives when asked which one is more liberal. We recommend selecting this option if you are estimating liberal-conservative ideology."),
    ui.tooltip(ui.input_checkbox("use_custom_prompts", "Use custom pairwise comparison prompts", width="400px"), "If not checked, uses a default pairwise comparison prompt that compares politicians on liberal-conservative ideology."),
    # Use unidirectional comparisons and custom prompts
    ui.panel_conditional(
        "!input.bidirectional_comparisons && input.use_custom_prompts",
        ui.tooltip(ui.input_text_area("prompt_unidirectional", "Pairwise comparison prompt", width="800px"), "Write your pairwise comparison prompt here. You can use any of the following Python string placeholders: {name0} (name of the first politician in pairwise comparison); {name1} (name of the second politician in pairwise comparison); {congress_number0} (congress number of first politician); {congress_number1} (congress number of second politician); {chamber0} (chamber of the first politician); {chamber1} (chamber of the second politician); {state0} (state abbrev. of the first politician); {state1} (state abbrev. of the second politician); {politician_type} (what to call politicians in pairwise comparisons, if set above; defaults to \"senator\", \"representative\", or \"politician\" if the Senate, House, or Both were selected as the chamber)")
    ),
    # Use bidirectional comparisons and custom prompts
    ui.panel_conditional(
        "input.bidirectional_comparisons && input.use_custom_prompts",
        ui.tooltip(ui.input_text_area("liberal_direction_prompt", "Liberal direction pairwise comparison prompt", width="800px"), "Write your liberal direction pairwise comparison prompt here. This will be used to compare Democrats against Democrats/Republicans/Independents. You can use any of the following Python string placeholders: {name0} (name of the first politician in pairwise comparison); {name1} (name of the second politician in pairwise comparison); {congress_number0} (congress number of first politician); {congress_number1} (congress number of second politician); {chamber0} (chamber of the first politician); {chamber1} (chamber of the second politician); {state0} (state abbrev. of the first politician); {state1} (state abbrev. of the second politician); {politician_type} (what to call politicians in pairwise comparisons, if set above; defaults to \"senator\", \"representative\", or \"politician\" if the Senate, House, or Both were selected as the chamber)"),
        ui.tooltip(ui.input_text_area("conservative_direction_prompt", "Conservative direction pairwise comparison prompt", width="800px"), "Write your conservative direction pairwise comparison prompt here. This will be used to compare Republicans against Republicans. Any of the same Python string placeholders can be used as previously described."),
    )
)

## Extraction Prompt Configuration Card
extraction_prompt_config_card = ui.card(
    ui.tooltip(ui.card_header("Extraction Prompt Configuration"), "You can customize the extraction prompt here. The extraction prompt is a follow up prompt that extracts the name of the preferred politician in each pairwise comparison without any additional explanation."),
    ui.tooltip(ui.input_checkbox("use_custom_extraction_prompts", "Use custom extraction prompts", width="400px"), "If not checked, uses a default extraction prompt."),
    # Use unidirectional comparisons and custom prompts
    ui.panel_conditional(
        "!input.bidirectional_comparisons && input.use_custom_extraction_prompts",
        ui.tooltip(ui.input_text_area("extraction_prompt_unidirectional", "Extraction prompt", width="800px"), "Write your extraction prompt here. Any of the same Python string placeholders can be used as previously described.")
    ),
    # Use bidirectional comparisons and custom prompts
    ui.panel_conditional(
        "input.bidirectional_comparisons && input.use_custom_extraction_prompts",
        ui.tooltip(ui.input_text_area("liberal_extraction_prompt", "Liberal direction extraction prompt", width="800px"), "Write your liberal direction extraction prompt here. Any of the same Python string placeholders can be used as previously described."),
        ui.tooltip(ui.input_text_area("conservative_extraction_prompt", "Conservative direction extraction prompt", width="800px"), "Write your conservative direction extraction prompt here. Any of the same Python string placeholders can be used as previously described.")
    ),
    ui.tooltip(ui.input_checkbox("scale_increasing_intensity", "Preferred legislators in pairwise comparisons placed on right", width="600px"), "As unidimensional scales are rotation invariant, this option sets the direction of the scale. Checking this box means more frequently preferred politicians in pairwise comparisons are placed to the right of the scale (have larger numbers). If you are using bidirectional comparisons, this option defaults to the direction of the liberal direction prompt. We recommend NOT checking this box if using the default liberal-conservative ideology prompts, such that more liberal politicians will be on the left.")
)

console_output_card = ui.card(
    ui.card_header("Console Output"),
    ui.output_text_verbatim("console_output_verbatim", placeholder=True)
)

run_button = ui.input_action_button("run_button", "Run pairwise comparisons", class_="btn-primary mt-3")

download_button = ui.output_ui("download_ui")

app_ui = ui.page_fluid(
    ui.h2("Generating Pairwise Comparisons Using an LLM"),
    ui.layout_columns(model_config_card, voteview_config_card, col_widths=[6, 6]),
    ui.layout_columns(prompt_config_card, extraction_prompt_config_card, col_widths=[6, 6]),
    ui.layout_columns(console_output_card, col_widths=[12]),
    ui.layout_columns(run_button, col_widths=[4]),
    ui.layout_columns(download_button, col_widths=[4])
)

class StreamToReactive:
    def __init__(self, reactive_val):
        self.reactive_val = reactive_val

    def write(self, text):
        current_value = self.reactive_val.get()
        self.reactive_val.set(current_value + text)

    def flush(self):
        # This method is required for file-like objects
        pass


@contextmanager
def reactive_console_output(reactive_val):
    old_stdout = sys.stdout
    sys.stdout = StreamToReactive(reactive_val)
    try:
        yield
    finally:
        sys.stdout = old_stdout

def server(input, output, session):
    results_df_store = reactive.Value(None)
    console_log = reactive.Value("")

    async def do_run():
        console_log.set("")

        progress_info = {"bar": None, "phase": ""}

        async def batch_progress_callback(completed, total):
            if progress_info["bar"]:
                progress_info["bar"].set(value=completed, message=f"{progress_info['phase']}: {completed}/{total}")
            await asyncio.sleep(0.001)

        with reactive_console_output(console_log):
            try:
                if not input.api_key():
                    ui.modal_show(ui.modal("Please enter API key.", easy_close=True))
                    return 

                seed_value = None
                if input.seed() != "":
                    try:
                        seed_value = int(input.seed())
                    except ValueError:
                        ui.modal_show(ui.modal("Seed must be an integer.", easy_close=True))
                        return
                    
                num_sample = None
                if input.num_sample() != "":
                    try:
                        num_sample = int(input.num_sample())
                    except ValueError:
                        ui.modal_show(ui.modal("Number of samples must be an integer.", easy_close=True))
                        return
                
                results_df_store.set(None)
                    
                # Read the API key and setup client 
                client_kwargs = dict(api_key=input.api_key())
                if input.provider() == 'deepinfra':
                    client_kwargs['base_url'] = DEEPINFRA_BASE_URL
                client = AsyncOpenAI(**client_kwargs)

                prompt = input.prompt_unidirectional() if input.prompt_unidirectional() != "" else None
                liberal_direction_prompt = input.liberal_direction_prompt() if input.liberal_direction_prompt() != "" else None
                conservative_direction_prompt = input.conservative_direction_prompt() if input.conservative_direction_prompt() != "" else None

                extraction_prompt = input.extraction_prompt_unidirectional() if input.extraction_prompt_unidirectional() != "" else None
                liberal_extraction_prompt = input.liberal_extraction_prompt() if input.liberal_extraction_prompt() != "" else None
                conservative_extraction_prompt = input.conservative_extraction_prompt() if input.conservative_extraction_prompt() != "" else None

                common_lamp_kwargs = {
                    "client": client,
                    "model": input.model(),
                    "voteview_df": input.custom_voteview_data(),
                    "unidirectional": not input.bidirectional_comparisons(),
                    "prompt": prompt,
                    "liberal_direction_prompt": liberal_direction_prompt,
                    "conservative_direction_prompt": conservative_direction_prompt,
                    "extraction_prompt": extraction_prompt,
                    "liberal_extraction_prompt": liberal_extraction_prompt,
                    "conservative_extraction_prompt": conservative_extraction_prompt,
                    "scale_increasing_intensity": not input.scale_increasing_intensity(),
                    "randomize_pairwise_order_seed": seed_value,
                    "sample_per_item": num_sample if num_sample != "" else None,
                    "concurrency": int(input.concurrency()), 
                    "temperature": input.temperature(),
                    "top_p": input.top_p(),
                    "progress_callback": batch_progress_callback
                }

                if input.use_own_voteview(): 
                    lamp_kwargs = {
                        **common_lamp_kwargs,
                        "voteview_df": input.custom_voteview_data(),
                        "politician_type": input.custom_politician_type(),
                        "canonical_names": input.use_canonical_names(),
                    }

                else:   
                    lamp_kwargs = {
                        **common_lamp_kwargs,
                        "congress_number": int(input.congress_numbers()),
                        "chamber": input.chamber(),
                        "politician_type": "senator" if input.chamber()=="S" else ("representative" if input.chamber()=="H" else "politician"),
                    }

                lamp = LaMPscores(**lamp_kwargs)

                lamp.create_matchups()
                total_comparisons = len(lamp.matchups_by_id_og)

                if lamp.unidirectional:
                    lamp.create_pairwise_comparison_prompt_ideology_unidirectional()
                    lamp.create_extraction_prompts_unidirectional()
                else:
                    lamp.create_pairwise_comparison_prompt_ideology_bidirectional()
                    lamp.create_extraction_prompts_bidirectional()

                with ui.Progress(min=0, max=total_comparisons) as progress:
                    progress_info["bar"] = progress
                    
                    # Pairwise comparisons
                    progress_info["phase"] = "Running pairwise comparisons"
                    progress.set(0, message=f"{progress_info['phase']}: 0/{total_comparisons}")
                    await lamp.run_pairwise_comparisons()
                    
                    # Extraction
                    progress_info["phase"] = "Extracting answers"
                    progress.set(0, message=f"{progress_info['phase']}: 0/{total_comparisons}")
                    await lamp.run_extraction()
                
                if lamp.unidirectional:
                    lamp.make_final_df_undirectional()
                else:
                    lamp.make_final_df_bidirectional()
                
                results_df = getattr(lamp, "matchup_results_df", None)

                if results_df is not None:
                    results_df_store.set(results_df)
                    print("\n\nRun complete. Results are ready for download.")
                else:
                    print("\n\nWARNING: Run finished, but no results were generated.")
        
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()

    @reactive.effect
    @reactive.event(input.run_button)
    async def _():
        await do_run()

    @render.text
    def console_output_verbatim():
        return console_log.get()

    @render.ui
    def download_ui():
        if results_df_store.get() is not None:
            return ui.download_button("download_results", "Download Pairwise Comparison Results", class_="btn-secondary mt-3")
        
    
    @render.download(filename="lamp_pairwise_comparisons.csv")
    def download_results():
        df = results_df_store.get()
        if df is None:
            return

        with io.StringIO() as buf:
            df.to_csv(buf, index=False)
            yield buf.getvalue().encode("utf-8")

app = App(app_ui, server)