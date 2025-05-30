import os, gget, json, gseapy, requests, anthropic, scanpy as sc, numpy as np, pandas as pd
from typing import Dict, Any, Optional
from verl.workers.agentic.biomni.llm import get_llm
from typing import Dict, Any, Optional, List, Union
import anthropic
import pickle

def get_rna_seq_archs4(gene_name: str, K: int = 10) -> str:
    """
    Given a gene name, this function returns the steps it performs and the max K transcripts-per-million (TPM)
    per tissue from the RNA-seq expression.
    
    Parameters:
    - gene_name (str): The gene name for which RNA-seq data is being fetched.
    - K (int): The number of tissues to return. Default is 10.

    Returns:
    - str: The steps performed and the result.
    """
    steps_log = f"Starting RNA-seq data fetch for gene: {gene_name} with K: {K}\n"

    try:
        # Fetch RNA-seq data using gget
        steps_log += "Fetching RNA-seq data using gget.archs4...\n"
        data = gget.archs4(gene_name, which="tissue")
        
        if data.empty:
            steps_log += f"No RNA-seq data found for the gene {gene_name}.\n"
            return steps_log
        
        # Create a readable output string
        steps_log += f"RNA-seq expression data for {gene_name} fetched successfully. Formatting the top {K} tissues:\n"
        readable_output = ""
        for index, row in data.iterrows():
            if index < K:
                tissue = row['id']
                median_tpm = row['median']
                readable_output += (
                    f"\nTissue: {tissue}\n"
                    f"  - Median TPM: {median_tpm}\n"
                )
            else:
                break
        
        steps_log += readable_output
        return steps_log
    
    except Exception as e:
        return f"An error occurred: {e}"

def get_gene_set_enrichment_analysis_supported_database_list() -> list:
    return gseapy.get_library_name()

def gene_set_enrichment_analysis(genes: list, top_k: int = 10, database: str = "ontology", background_list: list = None, plot: bool = False) -> str:
    """
    Perform enrichment analysis for a list of genes, with optional background gene set and plotting functionality.

    Parameters:
    - genes (list): List of gene symbols to analyze.
    - top_k (int): Number of top pathways to return. Default is 10.
    - database (str): User-friendly name of the database to use for enrichment analysis.
        Popular options include:
        - 'pathway'      (KEGG_2021_Human)
        - 'transcription'   (ChEA_2016)
        - 'ontology'     (GO_Biological_Process_2021)
        - 'diseases_drugs'  (GWAS_Catalog_2019)
        - 'celltypes'     (PanglaoDB_Augmented_2021)
        - 'kinase_interactions' (KEA_2015)
        You can use get_gene_set_enrichment_analysis_supported_database_list tool to get the list of supported databases.

    - background_list (list, optional): List of background genes to use for enrichment analysis.
    - plot (bool, optional): If True, generates a bar plot of the top K enrichment results.

    Returns:
    - str: The steps performed and the top K enrichment results.
    """

    steps_log = f"Starting enrichment analysis for genes: {', '.join(genes)} using {database} database and top_k: {top_k}\n"
    
    if background_list:
        steps_log += f"Using background list with {len(background_list)} genes.\n"
    
    try:
        # Perform enrichment analysis with or without background list
        steps_log += f"Performing enrichment analysis using gget.enrichr with the {database} database...\n"
        df = gget.enrichr(genes, database=database, background_list=background_list, plot=plot)
        
        # Limit to top K results
        steps_log += f"Filtering the top {top_k} enrichment results...\n"
        df = df.head(top_k)
        
        # Format the result
        output_str = ""
        for idx, row in df.iterrows():
            output_str += (
                f"Rank: {row['rank']}\n"
                f"Path Name: {row['path_name']}\n"
                f"P-value: {row['p_val']:.2e}\n"
                f"Z-score: {row['z_score']:.6f}\n"
                f"Combined Score: {row['combined_score']:.6f}\n"
                f"Overlapping Genes: {', '.join(row['overlapping_genes'])}\n"
                f"Adjusted P-value: {row['adj_p_val']:.2e}\n"
                f"Database: {row['database']}\n"
                "----------------------------------------\n"
            )
        
        steps_log += output_str


        return steps_log

    except Exception as e:
        return f"An error occurred: {e}"

