from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.messages.base import get_msg_title_repr
from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool
from typing import Any, ClassVar, Dict, List, Optional
import importlib
import json
import pickle
import enum
import requests
import os
import pandas as pd

def get_tool_decorated_functions(relative_path):
    import ast, os
    import importlib.util
    # Get the directory of the current file (__init__.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path from the relative path
    file_path = os.path.join(current_dir, relative_path)
    
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    tool_function_names = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'tool':
                    tool_function_names.append(node.name)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'tool':
                    tool_function_names.append(node.name)
    
    # Calculate the module name from the relative path
    package_path = os.path.relpath(file_path, start=current_dir)
    module_name = package_path.replace(os.path.sep, '.').rsplit('.', 1)[0]
    
    # Import the module and get the function objects
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    tool_functions = [getattr(module, name) for name in tool_function_names]
    
    return tool_functions

def pretty_print(message, printout = True):
    if isinstance(message, tuple):
        title = message
    else:
        if isinstance(message, list):
            title = json.dumps(message, indent=2)
            print(title)
        elif isinstance(message.content, list):
            title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"

            for i in message.content:
                if i['type'] == 'text':
                    title += f"\n{i['text']}\n"
                elif i['type'] == 'tool_use':
                    title += f"\nTool: {i['name']}"
                    title += f"\nInput: {i['input']}"
                elif i['type'] == 'thinking':
                    title += f"\nThinking: {i['thinking']}"
            if printout:
                print(f"{title}")
        else:
            title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"
            title += f"\n\n{message.content}"
            if printout:
                print(f"{title}")
    return title

class CustomBaseModel(BaseModel):
    api_schema: ClassVar[dict] = None  # Class variable to store api_schema
    
    # Add model_config with arbitrary_types_allowed=True
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def set_api_schema(cls, schema: dict):
        cls.api_schema = schema

    @classmethod
    def model_validate(cls, obj):
        try:
            return super().model_validate(obj)
        except (ValidationError, AttributeError) as e:
            if not cls.api_schema:
                raise e  # If no api_schema is set, raise original error

            error_msg = "Required Parameters:\n"
            for param in cls.api_schema['required_parameters']:
                error_msg += f"- {param['name']} ({param['type']}): {param['description']}\n"
            
            error_msg += "\nErrors:\n"
            for err in e.errors():
                field = err["loc"][0] if err["loc"] else "input"
                error_msg += f"- {field}: {err['msg']}\n"
            
            if not obj:
                error_msg += "\nNo input provided"
            else:
                error_msg += "\nProvided Input:\n"
                for key, value in obj.items():
                    error_msg += f"- {key}: {value}\n"
                
                missing_params = set(param['name'] for param in cls.api_schema['required_parameters']) - set(obj.keys())
                if missing_params:
                    error_msg += "\nMissing Parameters:\n"
                    for param in missing_params:
                        error_msg += f"- {param}\n"
            
            # # Create proper validation error structure
            raise ValidationError.from_exception_data(
                title="Validation Error",
                line_errors=[{
                    'type': 'value_error',
                    'loc': ('input',),
                    'input': obj,
                    "ctx": {
                        "error": error_msg,
                    },
                }]
            )

def safe_execute_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return str(e)
    return wrapper

def api_schema_to_langchain_tool(api_schema, mode = 'generated_tool', module_name = None):
    if mode == 'generated_tool':
        module = importlib.import_module('verl.workers.agentic.biomni.tool.generated_tool.' + api_schema['tool_name'] + '.api')
    elif mode == 'custom_tool':
        module = importlib.import_module("verl.workers.agentic." + module_name)

    api_function = getattr(module, api_schema['name'])
    api_function = safe_execute_decorator(api_function)

    # Define a mapping from string type names to actual Python type objects
    type_mapping = {
        'string': str,
        'integer': int,
        'boolean': bool,
        'pandas': pd.DataFrame,  # Use the imported pandas.DataFrame directly
        'str': str,
        'int': int,
        'bool': bool,
        'List[str]': List[str],
        'List[int]': List[int],
        'Dict': Dict,
        'Any': Any
    }

    # Create the fields and annotations
    annotations = {}
    for param in api_schema['required_parameters']:
        param_type = param['type']
        if param_type in type_mapping:
            annotations[param['name']] = type_mapping[param_type]
        else:
            # For types not in the mapping, try a safer approach than direct eval
            try:
                annotations[param['name']] = eval(param_type)
            except (NameError, SyntaxError):
                # Default to Any for unknown types
                annotations[param['name']] = Any

    fields = {
        param['name']: Field(description=param['description'])
        for param in api_schema['required_parameters']
    }

    # Create the ApiInput class dynamically
    ApiInput = type(
        "Input",
        (CustomBaseModel,),
        {
            '__annotations__': annotations,
            **fields
        }
    )
    # Set the api_schema
    ApiInput.set_api_schema(api_schema)

    # Create the StructuredTool
    api_tool = StructuredTool.from_function(
        func=api_function,
        name=api_schema['name'],
        description=api_schema['description'],
        args_schema=ApiInput,
        return_direct=True 
    )

    return api_tool

class ID(enum.Enum):
    ENTREZ = "Entrez"
    ENSEMBL = "Ensembl without version" # e.g. ENSG00000123374
    ENSEMBL_W_VERSION = "Ensembl with version" # e.g. ENSG00000123374.10 (needed for GTEx)


def get_gene_id(gene_symbol: str, id_type: ID):
    '''
    Get the ID for a gene symbol. If no match found, returns None
    '''
    if id_type == ID.ENTREZ:
        return _get_gene_id_entrez(gene_symbol)
    elif id_type == ID.ENSEMBL:
        return _get_gene_id_ensembl(gene_symbol)
    elif id_type == ID.ENSEMBL_W_VERSION:
        return _get_gene_id_ensembl_with_version(gene_symbol)
    else:
        raise ValueError(f"ID type {id_type} not supported")


def _get_gene_id_entrez(gene_symbol: str):
    '''
    Get the Entrez ID for a gene symbol. If no match found, returns None
    e.g. 1017 (CDK2)
    '''
    api_call = f'https://mygene.info/v3/query?species=human&q=symbol:{gene_symbol}'
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        return response_json["hits"][0]["entrezgene"]

def _get_gene_id_ensembl(gene_symbol):
    '''
    Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374
    '''
    api_call = f'https://mygene.info/v3/query?species=human&fields=ensembl&q=symbol:{gene_symbol}'
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        ensembl = response_json["hits"][0]["ensembl"]
        if isinstance(ensembl, list):
            return ensembl[0]["gene"] # Sometimes returns a list, for example RNH1 (first elem is on chr11, second is on scaffold_hschr11)
        else:
            return ensembl["gene"]
    
def _get_gene_id_ensembl_with_version(gene_symbol):
    '''
    Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374.10
    '''
    api_base = f"https://gtexportal.org/api/v2/reference/gene"
    params = {
        "geneId": gene_symbol
    }
    response_json = requests.get(api_base, params=params).json()

    if len(response_json["data"]) == 0:
        return None
    else:
        return response_json["data"][0]["gencodeId"]


def save_pkl(f, filename):
    with open(filename, 'wb') as file:
        pickle.dump(f, file)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}

def color_print(text, color="blue"):
    color_str = _TEXT_COLOR_MAPPING[color]
    print(f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m")


class PromptLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        for message in messages[0]:
            color_print(message.pretty_repr(), color="green")

class NodeLogger(BaseCallbackHandler):

    def on_llm_end(self, response, **kwargs): # response of type LLMResult
        for generations in response.generations: # response.generations of type List[List[Generations]] becuase "each input could have multiple candidate generations"
            for generation in generations:
                generated_text = generation.message.content
                #token_usage = generation.message.response_metadata["token_usage"]
                color_print(generated_text, color="yellow")

    def on_agent_action(self, action, **kwargs):
        color_print(action.log, color="pink")

    def on_agent_finish(self, finish, **kwargs):
        color_print(finish, color="red")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        color_print(f"Calling {tool_name} with inputs: {input_str}", color="pink")

    def on_tool_end(self, output, **kwargs):
        output = str(output)
        color_print(output, color="blue")

def check_or_create_path(path=None):
    # Set a default path if none is provided
    if path is None:
        path = os.path.join(os.getcwd(), 'tmp_directory')
    
    # Check if the path exists
    if not os.path.exists(path):
        # If it doesn't exist, create the directory
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists at: {path}")

    return path


def langchain_to_gradio_message(message):
    
    # Build the title and content based on the message type
    if isinstance(message.content, list):
        # For a message with multiple content items (like text and tool use)
        gradio_messages = []
        for item in message.content:
            gradio_message = {
                "role": "user" if message.type == "human" else "assistant",
                "content": "",
                "metadata": {}
                }

            if item['type'] == 'text':
                item['text'] = item['text'].replace('<think>', '\n')
                item['text'] = item['text'].replace('</think>', '\n')
                gradio_message["content"] += f"{item['text']}\n"
                gradio_messages.append(gradio_message)
            elif item['type'] == 'tool_use':
                if item['name'] == 'run_python_repl':
                    gradio_message["metadata"]["title"] = f"🛠️ Writing code..."
                    #input = "```python {code_block}```\n".format(code_block=item['input']["command"])
                    gradio_message["metadata"]['log'] = f"Executing Code block..."
                    gradio_message['content'] = f"##### Code: \n ```python \n {item['input']['command']} \n``` \n"
                else:
                    gradio_message["metadata"]["title"] = f"🛠️ Used tool ```{item['name']}```"
                    to_print = ';'.join([i + ': ' + str(j) for i,j in item['input'].items()])
                    gradio_message["metadata"]['log'] = f"🔍 Input -- {to_print}\n"
                gradio_message["metadata"]['status'] = "pending"
                gradio_messages.append(gradio_message)

    else:
        gradio_message = {
        "role": "user" if message.type == "human" else "assistant",
        "content": "",
        "metadata": {}
        }
        print(message)
        content = message.content
        content = content.replace('<think>', '\n')
        content = content.replace('</think>', '\n')
        content = content.replace('<solution>', '\n')
        content = content.replace('</solution>', '\n')
        
        gradio_message["content"] = content
        gradio_messages = [gradio_message]
    return gradio_messages


# parse_hpo_obo function was removed as it was unused

# Updated library_content as a dictionary with detailed descriptions
library_content_dict = {
    # === PYTHON PACKAGES ===
    
    # Core Bioinformatics Libraries (Python)
    "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
    #"biom-format": "[Python Package] The Biological Observation Matrix (BIOM) format is designed for representing biological sample by observation contingency tables with associated metadata.",
    "scanpy": "[Python Package] A scalable toolkit for analyzing single-cell gene expression data, specifically designed for large datasets using AnnData.",
    "scikit-bio": "[Python Package] Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
    #"anndata": "[Python Package] A Python package for handling annotated data matrices in memory and on disk, primarily used for single-cell genomics data.",
    #"mudata": "[Python Package] A Python package for multimodal data storage and manipulation, extending AnnData to handle multiple modalities.",
    #"pyliftover": "[Python Package] A Python implementation of UCSC liftOver tool for converting genomic coordinates between genome assemblies.",
    #"biopandas": "[Python Package] A package that provides pandas DataFrames for working with molecular structures and biological data.",
    #"biotite": "[Python Package] A comprehensive library for computational molecular biology, providing tools for sequence analysis, structure analysis, and more.",
    
    # Genomics & Variant Analysis (Python)
    "gget": "[Python Package] A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
    #"lifelines": "[Python Package] A complete survival analysis library for fitting models, plotting, and statistical tests.",
    #"scvi-tools": "[Python Package] A package for probabilistic modeling of single-cell omics data, including deep generative models.",
    "gseapy": "[Python Package] A Python wrapper for Gene Set Enrichment Analysis (GSEA) and visualization.",
    "cellxgene-census": "[Python Package] A tool for accessing and analyzing the CellxGene Census, a collection of single-cell datasets.",

    # Data Science & Statistical Analysis (Python)
    "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
    "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
    "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
    "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
    "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
    "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
    "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
    #"umap-learn": "[Python Package] Uniform Manifold Approximation and Projection, a dimension reduction technique.",
    #"faiss-cpu": "[Python Package] A library for efficient similarity search and clustering of dense vectors.",
    
    # General Bioinformatics & Computational Utilities (Python)
    #"tiledb": "[Python Package] A powerful engine for storing and analyzing large-scale genomic data.",
    #"tiledbsoma": "[Python Package] A library for working with the SOMA (Stack of Matrices) format using TileDB.",
    #"h5py": "[Python Package] A Python interface to the HDF5 binary data format, allowing storage of large amounts of numerical data.",
    #"tqdm": "[Python Package] A fast, extensible progress bar for loops and CLI applications.",
    #"joblib": "[Python Package] A set of tools to provide lightweight pipelining in Python, including transparent disk-caching and parallel computing.",
    "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
    "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
    #"scikit-image": "[Python Package] A collection of algorithms for image processing in Python.",
    "pymed": "[Python Package] A Python library for accessing PubMed articles.",
    "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
    "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar.",
    
    #"mageck": "[Python Package] Analysis of CRISPR screen data.",
    #"igraph": "[Python Package] Network analysis and visualization.",
    #"pyscenic": "[Python Package] Analysis of single-cell RNA-seq data and gene regulatory networks.",
}


def textify_api_dict(api_dict):
    """Convert a nested API dictionary to a nicely formatted string."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")
            
            # Process required parameters
            req_params = method.get('required_parameters', [])
            if req_params:
                lines.append("  Required Parameters:")
                for param in req_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")
            
            # Process optional parameters
            opt_params = method.get('optional_parameters', [])
            if opt_params:
                lines.append("  Optional Parameters:")
                for param in opt_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")
            
            
            lines.append("")  # Empty line between methods
        lines.append("")  # Extra empty line after each category

    return "\n".join(lines)

# Data lake dictionary with detailed descriptions
data_lake_dict = {
    "affinity_capture-ms.parquet": "Protein-protein interactions detected via affinity capture and mass spectrometry.",
    "affinity_capture-rna.parquet": "Protein-RNA interactions detected by affinity capture.",
    #"BindingDB_All_202409.tsv": "Measured binding affinities between proteins and small molecules for drug discovery.",
    #"broad_repurposing_hub_molecule_with_smiles.parquet": "Molecules from Broad Institute's Drug Repurposing Hub with SMILES annotations.",
    #"broad_repurposing_hub_phase_moa_target_info.parquet": "Drug phases, mechanisms of action, and target information from Broad Institute.",
    "co-fractionation.parquet": "Protein-protein interactions from co-fractionation experiments.",
    #"Cosmic_Breakpoints_v101_GRCh38.csv": "Genomic breakpoints associated with cancers from COSMIC database.",
    "Cosmic_CancerGeneCensusHallmarksOfCancer_v101_GRCh38.parquet": "Hallmarks of cancer genes from COSMIC.",
    #"Cosmic_CancerGeneCensus_v101_GRCh38.parquet": "Census of cancer-related genes from COSMIC.",
    #"Cosmic_ClassificationPaper_v101_GRCh38.parquet": "Cancer classifications and annotations from COSMIC.",
    #"Cosmic_Classification_v101_GRCh38.parquet": "Classification of cancer types from COSMIC.",
    #"Cosmic_CompleteCNA_v101_GRCh38.tsv.gz": "Complete copy number alterations data from COSMIC.",
    #"Cosmic_CompleteDifferentialMethylation_v101_GRCh38.tsv.gz": "Differential methylation patterns from COSMIC.",
    #"Cosmic_CompleteGeneExpression_v101_GRCh38.tsv.gz": "Gene expression data across cancers from COSMIC.",
    #"Cosmic_Fusion_v101_GRCh38.csv": "Gene fusion events from COSMIC.",
    "Cosmic_Genes_v101_GRCh38.parquet": "List of genes associated with cancer from COSMIC.",
    #"Cosmic_GenomeScreensMutant_v101_GRCh38.tsv.gz": "Genome screening mutations from COSMIC.",
    #"Cosmic_MutantCensus_v101_GRCh38.csv": "Catalog of cancer-related mutations from COSMIC.",
    #"Cosmic_ResistanceMutations_v101_GRCh38.parquet": "Resistance mutations related to therapeutic interventions from COSMIC.",
    #"czi_census_datasets_v4.parquet": "Datasets from the Chan Zuckerberg Initiative's Cell Census.",
    "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
    "dosage_growth_defect.parquet": "Gene dosage changes affecting growth.",
    #"enamine_cloud_library_smiles.pkl": "Compounds from Enamine REAL library with SMILES annotations.",
    "genebass_missense_LC_filtered.pkl": "Filtered missense variants from GeneBass.",
    "genebass_pLoF_filtered.pkl": "Predicted loss-of-function variants from GeneBass.",
    "genebass_synonymous_filtered.pkl": "Filtered synonymous variants from GeneBass.",
    "gene_info.parquet": "Comprehensive gene information.",
    "genetic_interaction.parquet": "Genetic interactions between genes.",
    "go-plus.json": "Gene ontology data for functional gene annotations.",
    "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
    "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
    #"hp.obo": "Official HPO release in obographs format",
    "marker_celltype.parquet": "Cell type marker genes for identification.",
    #"McPAS-TCR.parquet": "T-cell receptor sequences and specificity data from McPAS database.",
    "miRDB_v6.0_results.parquet": "Predicted microRNA targets from miRDB.",
    "miRTarBase_microRNA_target_interaction.parquet": "Experimentally validated microRNA-target interactions from miRTarBase.",
    "miRTarBase_microRNA_target_interaction_pubmed_abtract.txt": "PubMed abstracts for microRNA-target interactions in miRTarBase.",
    "miRTarBase_MicroRNA_Target_Sites.parquet": "Binding sites of microRNAs on target genes from miRTarBase.",
    "mousemine_m1_positional_geneset.parquet": "Positional gene sets from MouseMine.",
    "mousemine_m2_curated_geneset.parquet": "Curated gene sets from MouseMine.",
    "mousemine_m3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MouseMine.",
    "mousemine_m5_ontology_geneset.parquet": "Ontology-based gene sets from MouseMine.",
    "mousemine_m8_celltype_signature_geneset.parquet": "Cell type signature gene sets from MouseMine.",
    "mousemine_mh_hallmark_geneset.parquet": "Hallmark gene sets from MouseMine.",
    "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
    "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
    "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
    "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
    "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
    "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
    "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
    "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
    "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
    "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
    "omim.parquet": "Genetic disorders and associated genes from OMIM.",
    "proteinatlas.tsv": "Protein expression data from Human Protein Atlas.",
    "proximity_label-ms.parquet": "Protein interactions via proximity labeling and mass spectrometry.",
    "reconstituted_complex.parquet": "Protein complexes reconstituted in vitro.",
    "synthetic_growth_defect.parquet": "Synthetic growth defects from genetic interactions.",
    "synthetic_lethality.parquet": "Synthetic lethal interactions.",
    "synthetic_rescue.parquet": "Genetic interactions rescuing phenotypes.",
    "two-hybrid.parquet": "Protein-protein interactions detected by yeast two-hybrid assays.",
    #"variant_table.parquet": "Annotated genetic variants table.",
    #"Virus-Host_PPI_P-HIPSTER_2020.parquet": "Virus-host protein-protein interactions from P-HIPSTER.",
    #"txgnn_name_mapping.pkl": "Name mapping for TXGNN.",
    #"txgnn_prediction.pkl": "Prediction data for TXGNN."
}
