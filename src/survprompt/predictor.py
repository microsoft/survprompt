import os
import logging
import math
import re
import ast
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod
import asyncio
import aiolimiter
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
import openai
from tqdm.asyncio import tqdm_asyncio

from survprompt.configs.predictor_config import PredictorConfig
from survprompt.data.utils import FEATURE_TYPE_TO_COLS

def format_full_prompt(system_prompt: str,
                       user_prompt: str,
                       supports_system_prompt: bool) -> List[Dict[str, str]]:
    """Returns a list of dictionaries with the system and user prompts."""
    if supports_system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        return [
            {"role": "user", "content": '\n'.join([system_prompt, user_prompt])},
        ]

def extract_prediction(cfg: PredictorConfig, response: ChatCompletion) ->  Dict[str, Any]:
    """Extract the prediction from the response, based on the prompting task.

    Args:
    - cfg: PredictorConfig, the predictor configuration
    - response: Dict[str, Any], the response from the model

    Returns:
    - Dict[str, Any], containing predictions and full response text
    """
    try:
        response_text = response.choices[0].message.content
        if cfg.prompting_task == 'TTE_OS':
            return extract_prediction_TTE_OS(response_text, cfg)
        if cfg.prompting_task == 'SURV_PROB':
            # Extract the time points and survival probabilities from the response
            return extract_prediction_SURV_PROB(response_text, cfg)
    except Exception:
        return None

def extract_prediction_TTE_OS(response_text: str, cfg: PredictorConfig):
    if cfg.outcome_text in response_text:
        try:
            #num_days = int(response_text.split(f"{cfg.outcome_text}:")[1].split('\n')[0].strip())
            num_days = int(re.search(r'\d+', response_text.split(f"{cfg.outcome_text}:")[1]).group())
        except ValueError:
            num_days = None
    else:
        try:
            num_days = int(response_text)
        except ValueError:
            num_days = None

    return {'num_days': num_days, 'response_text': response_text}


def extract_prediction_SURV_PROB(response_text: str, cfg: PredictorConfig):
    try:
        time_probs = ast.literal_eval(re.search(r"\[\s*(?:\(\s*\d*\.?\d+\s*,\s*\d*\.?\d+\s*\)\s*,?\s*)+\]", response_text).group(0))
    except (ValueError, AttributeError, IndexError, SyntaxError):
        try:
            time_probs = ast.literal_eval(response_text)
        except (ValueError, AttributeError, IndexError, SyntaxError):
            time_probs = None
    try:
        time = [t[0] for t in time_probs]
        prob = [t[1] for t in time_probs]
    except (ValueError, TypeError, IndexError):
        time = None
        prob = None
            
    return {
        'time': time,
        'prob': prob,
        'response_text': response_text
    }

class VignetteCreator(ABC):
    """
    Abstract class for creating clinical vignettes from variables and values.
    """
    def __init__(self, variable2value: Dict[str, str]) -> None:
        self.variable2value = variable2value

    @abstractmethod
    def create_clinical_vignette(self, variable2value: Dict[str, str]) -> str:
        """Create a clinical vignette from the variables and values.
        Args:
        - variable2value: Dict[str, str], the variables and values
        Returns:
        - vignette: str, the clinical vignette
        """
        pass


class CancerVignette(VignetteCreator):
    def __init__(self, variable2value: Dict[str, str]) -> None:
        super().__init__(variable2value)
        self.variable2value = variable2value

    def create_clinical_vignette(self, variable2value: Dict[str, str]) -> str:
        """
        'path': Pathology
        'genomics': Genomics
        'treatment': Treatment
        'demographics': Demographics
        'met': Tumour sites (metastases and locations)
        'stage': Stage and Progression 
        'lab': Tumour markers (markers often measured as part of lab or biomarker tests)
        """
        sentences = []

        # Demographics
        sentences.append(self._create_text_demographic(variable2value))

        # Diagnosis
        sentences.append(self._create_text_cancer_info(variable2value))

        # Radiology (stage and metastasis)
        sentences.append(self._create_text_radiology(variable2value))

        # Pathology
        sentences.append(self._create_text_pathology(variable2value))

        # Genomics
        sentences.append(self._create_text_genomics(variable2value))

        vignette = " ".join(sentences)
    
        return vignette
    

    def _feature_exists(self, feature_type: str) -> bool:
        """
        Check if any column for a given feature_type is in self.variable2value.
        """
        feature_names = FEATURE_TYPE_TO_COLS[feature_type]
        return any(feature in self.variable2value for feature in feature_names)


    def _create_text_demographic(self, variable2value: Dict[str, str]) -> str:
        '''
        'demographics': ['AGE','MALE','WHITE','ASIAN','BLACK','SMOKER'],
        '''

        if not self._feature_exists('demographics'):
            return ""
        
        text = "The patient is a "
    
        if "AGE" in variable2value:
            text += f"{variable2value['AGE']} years old "

        if "WHITE" in variable2value and variable2value["WHITE"] == 1:
            text += "White "
        elif "ASIAN" in variable2value and variable2value["ASIAN"] == 1:
            text += "Asian "
        elif "BLACK" in variable2value and variable2value["BLACK"] == 1:
            text += "Black "
        else:
            pass

        if "MALE" in variable2value:
            sex = "male" if variable2value["MALE"] == 1 else "female"
            text += sex

        if "SMOKER" in variable2value and not math.isnan(variable2value["SMOKER"]):
            smoker = " with" if variable2value["SMOKER"] == 1 else " without"
            text += smoker + " a history of smoking"

        text += "."

        return text


    def _create_text_cancer_info(self, variable2value: Dict[str, str]) -> str:
        '''
        'stage': ['STAGE 1','STAGE 2','STAGE 3','STAGE 4','STAGE_IV_DX','STAGE_I-III_NOPROG','STAGE_I-III_PROG','progressed']
        'path': ['Gleason','HAS_Gleason','ADENOCARCINOMA','SQUAMOUS','PDL1','HAS_PDL1','HR','HER2', 'RECTAL','ASCENDING','CECUM','NONADENOCARCINOMA','MUCINOUS','MSI_OR_dMMR','HAS_MSI_OR_dMMR']
        'treatment': ['ANY_PRIOR_TX']
        '''
        
        text = "The patient has been diagnosed with "

        # Staging
        if self._feature_exists('stage'):
            if "STAGE 1" in variable2value and variable2value["STAGE 1"] == 1:
                text += "stage 1 "
            elif "STAGE 2" in variable2value and variable2value["STAGE 2"] == 1:
                text += "stage 2 "
            elif "STAGE 3" in variable2value and variable2value["STAGE 3"] == 1:
                text += "stage 3 "
            elif "STAGE 4" in variable2value and variable2value["STAGE 4"] == 1:
                text += "stage 4 "
            else:
                text += ""

        # Cancer type, in pathology features
        cancer_type = []
        if "has_nsclc" in variable2value and variable2value["has_nsclc"] == 1:
            cancer_subtype = []
            if "SQUAMOUS" in variable2value and not math.isnan(variable2value["SQUAMOUS"]):
                if variable2value["SQUAMOUS"] == 1:
                    cancer_subtype.append("squamous cell")
                else:
                    cancer_subtype.append("non-squamous cell")
            if "ADENOCARCINOMA" in variable2value and variable2value["ADENOCARCINOMA"] == 1:
                cancer_subtype.append("adenocarcinoma")
            if "NONADENOCARCINOMA" in variable2value and variable2value["NONADENOCARCINOMA"] == 1:
                cancer_subtype.append("non-adenocarcinoma")
            if len(cancer_subtype) > 0 and self._feature_exists('path'):
                cancer_type.append(", ".join(cancer_subtype) + ' non-small cell lung cancer (NSCLC)')
            else:
                cancer_type.append("non-small cell lung cancer (NSCLC)")
        if "has_brca" in variable2value and variable2value["has_brca"] == 1:
            cancer_subtype = []
            if "ADENOCARCINOMA" in variable2value and variable2value["ADENOCARCINOMA"] == 1:
                cancer_subtype.append("adenocarcinoma")
            if "NONADENOCARCINOMA" in variable2value and variable2value["NONADENOCARCINOMA"] == 1:
                cancer_subtype.append("non-adenocarcinoma")
            if "HER2" in variable2value and variable2value["HER2"] == 1:
                cancer_subtype.append("HER2 positive")
            if "HR" in variable2value and variable2value["HR"] == 1:
                cancer_subtype.append("HR positive")
            if len(cancer_subtype) > 0 and self._feature_exists('path'):
                cancer_type.append(", ".join(cancer_subtype) + ' breast cancer')
            else:
                cancer_type.append("breast cancer")
        if "has_crc" in variable2value and variable2value["has_crc"] == 1:
            cancer_subtype = []
            if "ADENOCARCINOMA" in variable2value and variable2value["ADENOCARCINOMA"] == 1:
                cancer_subtype.append("adenocarcinoma")
            if "NONADENOCARCINOMA" in variable2value and variable2value["NONADENOCARCINOMA"] == 1:
                cancer_subtype.append("non-adenocarcinoma")
            if "MUCINOUS" in variable2value and variable2value["MUCINOUS"] == 1:
                cancer_subtype.append("mucinous")
            if "ASCENDING" in variable2value and variable2value["ASCENDING"] == 1:
                cancer_subtype.append("ascending colon")
            if "CECUM" in variable2value and variable2value["CECUM"] == 1:
                cancer_subtype.append("cecum")
            if "RECTAL" in variable2value and variable2value["RECTAL"] == 1:
                cancer_subtype.append("rectal")
            if len(cancer_subtype) > 0 and self._feature_exists('path'):
                cancer_type.append(", ".join(cancer_subtype) + ' colorectal cancer')
            else:
                cancer_type.append("colorectal cancer")
        if "has_prostate" in variable2value and variable2value["has_prostate"] == 1:
            cancer_subtype = []
            if "ADENOCARCINOMA" in variable2value and variable2value["ADENOCARCINOMA"] == 1:
                cancer_subtype.append("adenocarcinoma")
            if "NONADENOCARCINOMA" in variable2value and variable2value["NONADENOCARCINOMA"] == 1:
                cancer_subtype.append("non-adenocarcinoma")
            if len(cancer_subtype) > 0 and self._feature_exists('path'):
                cancer_type.append(", ".join(cancer_subtype) + ' prostate cancer')
            else:
                cancer_type.append("prostate cancer")
        if "has_panc" in variable2value and variable2value["has_panc"] == 1:
            cancer_subtype = []
            if "ADENOCARCINOMA" in variable2value and variable2value["ADENOCARCINOMA"] == 1:
                cancer_subtype.append("adenocarcinoma")
            if "NONADENOCARCINOMA" in variable2value and variable2value["NONADENOCARCINOMA"] == 1:
                cancer_subtype.append("non-adenocarcinoma")
            if len(cancer_subtype) > 0 and self._feature_exists('path'):
                cancer_type.append(", ".join(cancer_subtype) + ' pancreatic cancer')
            else:
                cancer_type.append("pancreatic cancer")
        if len(cancer_type) > 0:
            text += ", ".join(cancer_type) + "."
    
        # Prior treatment
        # 'treatment': set(['ANY_PRIOR_TX'])
        if self._feature_exists('treatment'):
            if "ANY_PRIOR_TX" in variable2value:
                if variable2value["ANY_PRIOR_TX"] == 1:
                    text += " The patient has received prior treatment."
                else:
                    text += " The patient has not received prior treatment."

        return text

    def _create_text_radiology(self, variable2value: Dict[str, str]) -> str:
        '''
        'stage': ['STAGE 1','STAGE 2','STAGE 3','STAGE 4','STAGE_IV_DX','STAGE_I-III_NOPROG','STAGE_I-III_PROG','progressed']
        'met': ['DMETS_DX_ADRENAL','DMETS_DX_BONE','DMETS_DX_BRAIN','DMETS_DX_LIVER','DMETS_DX_LUNG','DMETS_DX_LYMPH','DMETS_DX_PLEURA','DMETS_DX_OTHER']
        '''
        if not any([key in variable2value for key in ['STAGE_IV_DX','STAGE_I-III_NOPROG','STAGE_I-III_PROG']]) and not self._feature_exists('met'):
            return ""

        text = "A recent radiology report shows: "

        # Radiology
        if self._feature_exists('stage'):
            if "STAGE_IV_DX" in variable2value and variable2value["STAGE_IV_DX"] == 1:
                text += "Stage IV disease"
                if "progressed" in variable2value and variable2value["progressed"] == 1:
                    text += " with evidence of progression"
            elif "STAGE_I-III_NOPROG" in variable2value and variable2value["STAGE_I-III_NOPROG"] == 1:
                text += "Stage I-III disease without evidence of progression"
            elif "STAGE_I-III_PROG" in variable2value and variable2value["STAGE_I-III_PROG"] == 1:
                text += "Stage I-III disease with evidence of progression"

        # Metastasis
        if self._feature_exists('met'):
            if not any([variable2value["DMETS_DX_ADRENAL"] == 1,
                        variable2value["DMETS_DX_BONE"] == 1,
                        variable2value["DMETS_DX_BRAIN"] == 1,
                        variable2value["DMETS_DX_LIVER"] == 1,
                        variable2value["DMETS_DX_LUNG"] == 1,
                        variable2value["DMETS_DX_LYMPH"] == 1,
                        variable2value["DMETS_DX_PLEURA"] == 1,
                        variable2value["DMETS_DX_OTHER"] == 1]):
                text += " without evidence of metastasis."
            else:
                text += " with evidence of metastasis to: "
                mets_sites = []
                if variable2value["DMETS_DX_ADRENAL"] == 1:
                    mets_sites += ["adrenal glands"]
                if variable2value["DMETS_DX_BONE"] == 1:
                    mets_sites += ["bone"]
                if variable2value["DMETS_DX_BRAIN"] == 1:
                    mets_sites += ["brain"]
                if variable2value["DMETS_DX_LIVER"] == 1:
                    mets_sites += ["liver"]
                if variable2value["DMETS_DX_LUNG"] == 1:
                    mets_sites += ["lung"]
                if variable2value["DMETS_DX_LYMPH"] == 1:
                    mets_sites += ["lymph nodes"]
                if variable2value["DMETS_DX_PLEURA"] == 1:
                    mets_sites += ["pleura"]
                if variable2value["DMETS_DX_OTHER"] == 1:
                    mets_sites += ["other sites"]
                
                text += ", ".join(mets_sites) + "."
        else:
            text += "."
    
        return text

    def _create_text_pathology(self, variable2value: Dict[str, str]) -> str:
        '''
        'path': ['Gleason','HAS_Gleason','ADENOCARCINOMA','SQUAMOUS','PDL1','HAS_PDL1','HR','HER2', 'RECTAL','ASCENDING','CECUM','NONADENOCARCINOMA','MUCINOUS','MSI_OR_dMMR','HAS_MSI_OR_dMMR'],

        [i for i in variable2value if i in FEATURE_TYPE_TO_COLS['path']]
        '''
        if not self._feature_exists('path'):
            return ""

        # Pathology
        text = ""
        if "HAS_PDL1" in variable2value and variable2value["HAS_PDL1"] == 1:
            if "PDL1" in variable2value and variable2value["PDL1"] == 1:
                text += "PDL1 expression of 1% or higher."
            else:
                text += "PDL1 expression of less than 1%."
        else:
            text += "PDL1 expression not assessed."
        if "HAS_Gleason" in variable2value and variable2value["HAS_Gleason"] == 1:
            if "Gleason" in variable2value and not math.isnan(variable2value["Gleason"]):
                text += f" Gleason score of {variable2value['Gleason']}."
            else:
                text += " Gleason score not assessed."
        if "HAS_MSI_OR_dMMR" in variable2value and variable2value["HAS_MSI_OR_dMMR"] == 1:
            if "MSI_OR_dMMR" in variable2value and variable2value["MSI_OR_dMMR"] == 1:
                text += " Patient has MSI-H or dMMR."
        
        if len(text) > 0:
            return "Pathology report shows: " + text
        return ""

    def _create_text_genomics(self, variable2value: Dict[str, str]) -> str:
        """
        'genomics': ['KRAS', 'HRAS', 'RET', 'MET', 'GNAQ', 'PTEN', 'KIT', 'EGFR', 'FGFR1', 'FGFR2', 'FGFR3', 'PDGFRA', 'ERBB2', 'TP53', 'NRAS', 'NOTCH1', 'GNA11', 'CTNNB1', 'PIK3CA', 'IDH1', 'BRAF', 'ALK', 'AKT1']
        """
        if not self._feature_exists('genomics'):
            return ""

        # Genomics
        text = ""
        genomic_alterations = []
        genomic_nonalterations = []
        for gene in ['KRAS', 'HRAS', 'RET', 'MET', 'GNAQ', 'PTEN', 'KIT', 'EGFR', 'FGFR1', 'FGFR2', 'FGFR3', 'PDGFRA', 'ERBB2', 'TP53', 'NRAS', 'NOTCH1', 'GNA11', 'CTNNB1', 'PIK3CA', 'IDH1', 'BRAF', 'ALK', 'AKT1']:
            if gene in variable2value and variable2value[gene] == 1:
                genomic_alterations.append(gene)
            else:
                genomic_nonalterations.append(gene)
        if len(genomic_alterations) > 0 or len(genomic_nonalterations) > 0:
            text += " oncogenic alterations (mutations, copy number changes and/or structural variations) in the following genes: "
            if len(genomic_alterations) > 0:
                text += ", ".join(genomic_alterations) + "."
            else:
                text += "None."
        if len(genomic_nonalterations) > 0:
            text += " There are no oncogenic alterations in the following genes: "
            text += ", ".join(genomic_nonalterations) + "."

        if len(text) > 0:
            return "Genomic testing shows" + text
        return ""

    def _create_text_lab(self, variable2value: Dict[str, str]) -> str:
        """
        Tumour markers
        'lab': ['MAX_CA15-3','CA15-3','HAS_CA15-3', 'MAX_CEA','CEA','HAS_CEA', 'MAX_PSA','PSA','HAS_PSA', 'MAX_CA19-9','CA19-9','HAS_CA19-9']
        """
        if not self._feature_exists('lab'):
            return ""

        # Tumor-specific tests/markers
        text = ""
        if "HAS_CA15-3" in variable2value and variable2value["HAS_CA15-3"] == 1:
            if "CA15-3" in variable2value and not math.isnan(variable2value["CA15-3"]):
                text += f" CA15-3 level of {variable2value['CA15-3']}."
                if "MAX_CA15-3" in variable2value and not math.isnan(variable2value["MAX_CA15-3"]):
                    text += f" Maximum CA15-3 level of {variable2value['MAX_CA15-3']}."
            else:
                text += " CA15-3 level not assessed."
        if "HAS_CEA" in variable2value and variable2value["HAS_CEA"] == 1:
            if "CEA" in variable2value and not math.isnan(variable2value["CEA"]):
                text += f" CEA level of {variable2value['CEA']}."
                if "MAX_CEA" in variable2value and not math.isnan(variable2value["MAX_CEA"]):
                    text += f" Maximum CEA level of {variable2value['MAX_CEA']}."
            else:
                text += " CEA level not assessed."
        if "HAS_CA19-9" in variable2value and variable2value["HAS_CA19-9"] == 1:
            if "CA19-9" in variable2value and not math.isnan(variable2value["CA19-9"]):
                text += f" CA19-9 level of {variable2value['CA19-9']}."
                if "MAX_CA19-9" in variable2value and not math.isnan(variable2value["MAX_CA19-9"]):
                    text += f" Maximum CA19-9 level of {variable2value['MAX_CA19-9']}."
            else:
                text += " CA19-9 level not assessed."
        if "HAS_PSA" in variable2value and variable2value["HAS_PSA"] == 1:
            if "PSA" in variable2value and not math.isnan(variable2value["PSA"]):
                text += f" PSA level of {variable2value['PSA']}."
                if "MAX_PSA" in variable2value and not math.isnan(variable2value["MAX_PSA"]):
                    text += f" Maximum PSA level of {variable2value['MAX_PSA']}."
            else:
                text += " PSA level not assessed."

        if len(text) > 0:
            return "Lab report shows: " + text
        return ""


class OutcomePredictor(ABC):
    def __init__(
        self,
        cfg: PredictorConfig
    ) -> None:
        self.cfg = cfg
        self.client = None
        self._init_client()

    @abstractmethod
    def _init_client(self):
        pass

    def create_vignette(self, variable2value: Dict[str, str]) -> str:
        """Create a clinical vignette from the variables and values.

        Args:
        - variable2value: Dict[str, str], the variables and values

        Returns:
        - vignette: str, the clinical vignette
        """
        vignette_creator = CancerVignette(variable2value)
        vignette = vignette_creator.create_clinical_vignette(variable2value)

        return vignette 

    def format_example(self, vignette: str, outcome: str) -> str:
        """Format a clinical vignette and outcome as an example.

        Args:
        - vignette: str, the clinical vignette
        - outcome: str, the outcome
        - outcome_text: str, the outcome text

        Returns:
        - example: str, the formatted example
        """
        return f"Vignette: {vignette}\n{self.cfg.outcome_text}: {outcome}"

    def format_for_evaluation(self,
                              query_variable2value: Dict[str, str],
                              examples_variable2value: List[Dict[str, str]] = None,
                              outcomes = List[str]
                              ) -> Tuple[str, str]:
        """Format the clinical information as a prompt.

        Args:
        -variable2value: Dict[str, str], the variables and values for the query clinical vignette
        -examples_variable2value: List[Dict[str, str]], the variables and values for the in context examples
        -examples_outcomes: List containing outcome values for examples, formatted as a string

        Returns:
        - system_prompt: str, the system prompt for evaluation
        - user_prompt: str, the user prompt for evaluation
        """

        system_prompt = self.cfg.system_instructions.render()

        examples_formatted = ""
        if examples_variable2value is not None:
            examples_list = []
            for i, ex in enumerate(examples_variable2value):
                vignette = self.create_vignette(ex)
                ex_outcomes = list(outcomes[i].values())
                ex_outcomes = ex_outcomes[0] if len(ex_outcomes) == 1 else ex_outcomes
                example = self.format_example(vignette, ex_outcomes)
                examples_list.append(example)
            examples_formatted = "\n\n".join(examples_list)

        query_vignette = self.create_vignette(query_variable2value)

        user_prompt = self.cfg.user_instructions.render(
            examples_formatted=examples_formatted,
            clinical_vignette=query_vignette,
        )
        return system_prompt, user_prompt

    @abstractmethod
    def _predict_outcome_one(
        self,
        formatted_prompt: str,
    ) -> Dict[str, Dict[str, int]]:
        pass

   
class OpenAIOutcomePredictor(OutcomePredictor):
    def _init_client(self):
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") if os.getenv("AZURE_OPENAI_API_VERSION") is not None else "2024-08-01-preview"
        # Entra ID scope to request a token for. Override via AZURE_OPENAI_TOKEN_SCOPE
        # when the deployment sits behind a gateway with its own application ID URI.
        token_scope = os.getenv("AZURE_OPENAI_TOKEN_SCOPE", "https://cognitiveservices.azure.com/.default")
        if self.client is None and not self.cfg.use_async:
            if os.getenv("AZURE_OPENAI_API_KEY") is not None:
                self.client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version,
                )
            else:
                self.client = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_ad_token_provider=get_bearer_token_provider(
                        AzureCliCredential(),
                        token_scope,
                    ),
                )
        elif self.client is None and self.cfg.use_async:
            if os.getenv("AZURE_OPENAI_API_KEY") is not None:
                self.client = AsyncAzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    max_retries=self.cfg.max_retries,
                )
            else:
                self.client = AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_ad_token_provider=get_bearer_token_provider(
                        AzureCliCredential(),
                        token_scope,
                    ),
                    max_retries=self.cfg.max_retries,
                )

    def _predict_outcome_one(
        self,
        formatted_prompt: str,
    ) -> Dict[str, Any]:
        """Evaluate the candidate against the reference.

        Args:
        - formatted_prompt: str, the prompt for evaluation

        Returns:
        - Dict[str, Any], containing predictions and full response from the model
        """

        response = self.generate_openai_chat_completion(formatted_prompt)
        pred_res = extract_prediction(self.cfg, response)

        num_retries = 0
        if pred_res is None and self.cfg.max_retries > 0:
            while num_retries < self.cfg.max_retries:
                response = self.generate_openai_chat_completion(formatted_prompt)
                pred_res = extract_prediction(self.cfg, response)
                num_retries += 1
                if pred_res is not None:
                    break

        return pred_res
    
    def predict_outcome(
            self,
            query_pt_dicts: List[Dict[str, str]] | Dict[str, str],
            examples_pt_dicts: List[List[Dict[str, str]]] = None,
            examples_pt_outcomes: List[List[str]] = None,
        ) -> List[Dict[str, Dict[str, int]]]:
        """Evaluate the candidates against the references.

        Args:
        - query_pt_dicts: List[Dict[str, str]], list of patient dictionaries mapping features to values
        - examples_pt_dicts: List[List[Dict[str, str]]], list of patient dictionaries mapping features to values
        - examples_pt_outcomes: List[List[str]], list of outcomes for the examples

        Returns:
        - results: List[Dict[str, Dict[str, int]]], the evaluation results
        """
        if not isinstance(query_pt_dicts, list):
            query_pt_dicts = [query_pt_dicts]

        formatted_prompts = []
        if examples_pt_dicts is None:
            for pt_dict in query_pt_dicts:
                system_prompt, user_prompt = self.format_for_evaluation(pt_dict)
                formatted_prompts.append(format_full_prompt(system_prompt, user_prompt, not self.cfg.is_o1))
        else: # example references have been provided
            for query_pt_dict, examples_pt_dict, examples_outcomes in zip(query_pt_dicts, examples_pt_dicts, examples_pt_outcomes):
                system_prompt, user_prompt = self.format_for_evaluation(query_pt_dict, examples_pt_dict, examples_outcomes)
                formatted_prompts.append(format_full_prompt(system_prompt, user_prompt, not self.cfg.is_o1))

        results = []
        if not self.cfg.use_async:
            from tqdm import tqdm
            for formatted_prompt in tqdm(formatted_prompts, desc="Predicting outcomes"):
                result = self._predict_outcome_one(formatted_prompt)
                results.append(result)
        else:
            responses = asyncio.run(
                self.generate_openai_batch_chat_completion(formatted_prompts),
            )
            completion_dicts = [extract_prediction(self.cfg, response) for response in responses]
            
            num_retries = 0
            n_to_retry = sum([1 for r in completion_dicts if r is None])

            if self.cfg.max_retries > 0 and n_to_retry > 0:
                while num_retries < self.cfg.max_retries:
                    logging.warning(f"Found {n_to_retry} invalid predictions. Retrying...")
                    to_reprompt = []
                    for i, r in enumerate(responses):
                        if extract_prediction(self.cfg, r) is None:
                            to_reprompt.append(i)
                    new_formatted_prompts = [formatted_prompts[i] for i in to_reprompt]
                    if not self.cfg.use_async:
                        new_responses = [
                            self.generate_openai_chat_completion(p) for p in new_formatted_prompts
                        ]
                    else:
                        new_responses = asyncio.run(self.generate_openai_batch_chat_completion(new_formatted_prompts))
                    for i, r in zip(to_reprompt, new_responses):
                        completion_dicts[i] = extract_prediction(self.cfg, r)
                            
                    num_retries += 1
                    if all([r is not None for r in completion_dicts]):
                        break

            results = completion_dicts

        return results, formatted_prompts

    def generate_openai_chat_completion(
        self, formatted_prompt: List[Dict[str, str]]
    ) -> Dict[str, str]:
        # o models and gpt 5 don't support all parameters
        if ('o' in self.cfg.pred_engine) or ('gpt-5' in self.cfg.pred_engine):
            extra_kwargs = {}
            # reasoning_effort is only supported by gpt-5 models
            if 'gpt-5' in self.cfg.pred_engine and self.cfg.reasoning_effort != "none":
                extra_kwargs['reasoning_effort'] = self.cfg.reasoning_effort
            return self.client.chat.completions.create(
                model=self.cfg.pred_engine,
                messages=formatted_prompt,
                max_completion_tokens=self.cfg.max_tokens,
                **extra_kwargs,
            )
        return self.client.chat.completions.create(
            model=self.cfg.pred_engine,
            messages=formatted_prompt,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            top_p=self.cfg.top_p,
            frequency_penalty=self.cfg.frequency_penalty,
            presence_penalty=self.cfg.presence_penalty,
            stop=self.cfg.stop,
        )

    async def generate_openai_chat_completion_async(
        self,
        formatted_prompt: List[Dict[str, str]],
        limiter: aiolimiter.AsyncLimiter,
        **kwargs,
    ) -> Dict[str, str]:
        async with limiter:
            for trial_count in range(5):
                try:
                    # o models and gpt 5 don't support all parameters
                    if ('o' in self.cfg.pred_engine) or ('gpt-5' in self.cfg.pred_engine):
                        extra_kwargs = dict(kwargs)
                        # reasoning_effort is only supported by gpt-5 models
                        if 'gpt-5' in self.cfg.pred_engine and self.cfg.reasoning_effort != "none":
                            extra_kwargs['reasoning_effort'] = self.cfg.reasoning_effort
                        return await self.client.chat.completions.create(
                            model=self.cfg.pred_engine,
                            messages=formatted_prompt,
                            max_completion_tokens=self.cfg.max_tokens,
                            **extra_kwargs,
                        )
                    return await self.client.chat.completions.create(
                        model=self.cfg.pred_engine,
                        messages=formatted_prompt,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                        top_p=self.cfg.top_p,
                        frequency_penalty=self.cfg.frequency_penalty,
                        presence_penalty=self.cfg.presence_penalty,
                        stop=self.cfg.stop,
                        **kwargs,
                    )

                except openai.RateLimitError: #note, Async client is supposed to handle this, but unclear how, so re-handling here
                    sleep_time =  10 * (1 + trial_count**2)
                    logging.warning(
                        f"OpenAI API rate limit exceeded. Sleeping for {sleep_time} seconds."
                    )
                    await asyncio.sleep(sleep_time)
                except openai.AuthenticationError as e:
                    logging.warning(f"OpenAI authentication error: {e}")
                    break
                except Exception as e:
                    logging.warning(f"OpenAI error: {e}")
                    break

    async def generate_openai_batch_chat_completion(
        self,
        formatted_prompts: List,
        **kwargs
    ) -> List[Dict[str, Dict[str, int] | None]]:
        """Generate from OpenAI Chat Completion API.

        Args:
            formatted_prompts: List of formatted prompts generate from.
        Returns:
            List of generated responses.
        """
        limiter = aiolimiter.AsyncLimiter(self.cfg.requests_per_minute)
        async_responses = [
            self.generate_openai_chat_completion_async(
                formatted_prompt=p,
                limiter=limiter,
                **kwargs
            )
            for p in formatted_prompts
        ]
        responses = await tqdm_asyncio.gather(*async_responses, desc="Predicting outcomes")

        return responses
