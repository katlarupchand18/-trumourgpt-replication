"""
LDA Topic Modeling for TrumorGPT.
Implements Latent Dirichlet Allocation for health-related document corpus.
Sentence-level topic distributions as per paper Section III-B.1.
"""

import pickle
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.preprocessing.text import TextPreprocessor, PreprocessingConfig


@dataclass
class LDAConfig:
    """Configuration for LDA topic model."""
    num_topics: int = 20
    alpha: Union[str, float] = "auto"
    eta: Union[str, float] = "auto"
    iterations: int = 1000
    passes: int = 10
    random_state: int = 42
    min_df: int = 5
    max_df: float = 0.95

    # Corpus
    corpus_source: str = "pubmed_abstracts"
    corpus_path: str = "data/corpus/health_corpus.txt"
    model_path: str = "data/cache/lda_model"

    # Health topic identification
    health_topic_keywords: List[str] = field(default_factory=lambda: [
        "health", "medical", "medicine", "disease", "pandemic", "epidemic",
        "virus", "vaccine", "treatment", "therapy", "diagnosis", "symptom",
        "patient", "hospital", "doctor", "nurse", "clinical", "trial",
        "drug", "pharmaceutical", "fda", "cdc", "who", "public health",
        "covid", "coronavirus", "sars", "infection", "immunity", "antibody"
    ])
    health_topic_boost: float = 1.5  # Paper's α parameter

    # Preprocessing
    additional_stopwords: List[str] = field(default_factory=lambda: [
        "said", "say", "says", "according", "report", "reports", "study", "studies",
        "research", "researchers", "scientists", "experts", "officials", "according",
        "new", "year", "years", "time", "times", "week", "weeks", "month", "months",
        "day", "days", "today", "yesterday", "tomorrow", "last", "next", "first",
        "second", "third", "also", "however", "therefore", "thus", "hence", "since"
    ])


@dataclass
class TopicDistribution:
    """Topic distribution for a document/sentence."""
    topic_ids: np.ndarray  # Shape: (num_topics,)
    topic_probs: np.ndarray  # Shape: (num_topics,)
    dominant_topic: int
    dominant_prob: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_ids": self.topic_ids.tolist(),
            "topic_probs": self.topic_probs.tolist(),
            "dominant_topic": int(self.dominant_topic),
            "dominant_prob": float(self.dominant_prob),
        }


class HealthTopicIdentifier:
    """Identify health-related topics in LDA model."""

    def __init__(self, keywords: List[str], boost: float = 1.5):
        self.keywords = [k.lower() for k in keywords]
        self.boost = boost
        self._health_topic_ids: Optional[List[int]] = None

    def identify_health_topics(self, lda_model: LdaModel, top_n_words: int = 20) -> List[int]:
        """Identify which topics are health-related based on top words."""
        health_topics = []

        for topic_id in range(lda_model.num_topics):
            # Get top words for this topic
            top_words = lda_model.show_topic(topic_id, topn=top_n_words)
            word_scores = {word: score for word, score in top_words}

            # Check if any health keyword appears in top words
            health_score = 0.0
            for keyword in self.keywords:
                # Check for keyword in top words (partial matching)
                for word, score in top_words:
                    if keyword in word.lower() or word.lower() in keyword:
                        health_score += score

            if health_score > 0:
                health_topics.append(topic_id)
                logger.debug(f"Topic {topic_id} identified as health-related (score: {health_score:.4f})")

        self._health_topic_ids = health_topics
        logger.info(f"Identified {len(health_topics)} health-related topics: {health_topics}")
        return health_topics

    def get_topic_relevance(self, topic_dist: np.ndarray) -> np.ndarray:
        """
        Compute topic relevance scores R(v_i) as per paper Section III-C.
        R(v_i) = Σ(β_k * t_i,k) / Σ(β_k * t_j,k)
        where β_k = α (health_topic_boost) if topic k is health-related, else 1.
        """
        if self._health_topic_ids is None:
            # Default: uniform relevance
            return np.ones_like(topic_dist) / len(topic_dist)

        num_topics = len(topic_dist)
        beta = np.ones(num_topics)
        beta[self._health_topic_ids] = self.boost

        # Weighted topic distribution
        weighted = topic_dist * beta

        # Normalize
        total = weighted.sum()
        if total > 0:
            return weighted / total
        else:
            return np.ones(num_topics) / num_topics

    @property
    def health_topic_ids(self) -> List[int]:
        return self._health_topic_ids or []


class LDATopicModel:
    """
    LDA Topic Model for health domain.
    Trains on health corpus, provides sentence-level topic distributions.
    """

    def __init__(self, config: Optional[LDAConfig] = None):
        self.config = config or self._load_config()
        self.model: Optional[LdaModel] = None
        self.dictionary: Optional[corpora.Dictionary] = None
        self.health_identifier: Optional[HealthTopicIdentifier] = None
        self._preprocessor = TextPreprocessor(PreprocessingConfig(
            preserve_entities=False,
            preserve_negation=False,
            preserve_numerical=False,
            preserve_dates=False,
        ))

    def _load_config(self) -> LDAConfig:
        settings = get_settings()
        return LDAConfig(
            num_topics=settings.topic_modeling.num_topics,
            alpha=settings.topic_modeling.alpha,
            eta=settings.topic_modeling.eta,
            iterations=settings.topic_modeling.iterations,
            passes=settings.topic_modeling.passes,
            random_state=settings.topic_modeling.random_state,
            min_df=settings.topic_modeling.min_df,
            max_df=settings.topic_modeling.max_df,
            corpus_source=settings.topic_modeling.corpus_source,
            corpus_path=settings.topic_modeling.corpus_path,
            model_path=settings.topic_modeling.model_path,
            health_topic_keywords=settings.topic_modeling.health_topic_keywords,
            health_topic_boost=settings.topic_modeling.health_topic_boost,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for LDA."""
        # Simple tokenization - in production use spaCy
        tokens = text.lower().split()
        # Remove stopwords and short tokens
        stopwords = set(STOPWORDS) | set(self.config.additional_stopwords)
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2 and t.isalpha()]
        return tokens

    def prepare_corpus(self, texts: List[str]) -> List[List[str]]:
        """Prepare corpus for LDA training."""
        logger.info(f"Preparing corpus from {len(texts)} documents...")
        tokenized = [self._tokenize(text) for text in texts]

        # Filter empty documents
        tokenized = [t for t in tokenized if len(t) > 0]
        logger.info(f"Prepared {len(tokenized)} tokenized documents")
        return tokenized

    def train(self, texts: List[str], save: bool = True) -> None:
        """Train LDA model on corpus."""
        logger.info(f"Training LDA with {self.config.num_topics} topics...")

        # Prepare corpus
        tokenized = self.prepare_corpus(texts)

        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized)
        self.dictionary.filter_extremes(
            no_below=self.config.min_df,
            no_above=self.config.max_df
        )
        logger.info(f"Dictionary size: {len(self.dictionary)}")

        # Create BoW corpus
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in tokenized]

        # Train LDA
        self.model = LdaModel(
            corpus=bow_corpus,
            id2word=self.dictionary,
            num_topics=self.config.num_topics,
            alpha=self.config.alpha,
            eta=self.config.eta,
            iterations=self.config.iterations,
            passes=self.config.passes,
            random_state=self.config.random_state,
            eval_every=None,  # Disable perplexity evaluation for speed
        )

        # Identify health topics
        self.health_identifier = HealthTopicIdentifier(
            self.config.health_topic_keywords,
            self.config.health_topic_boost
        )
        self.health_identifier.identify_health_topics(self.model)

        if save:
            self.save()

        logger.info("LDA training completed")

    def get_topic_distribution(self, text: str) -> TopicDistribution:
        """Get topic distribution for a single text (sentence or document)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Tokenize
        tokens = self._tokenize(text)
        if not tokens:
            # Return uniform distribution
            uniform = np.ones(self.config.num_topics) / self.config.num_topics
            return TopicDistribution(
                topic_ids=np.arange(self.config.num_topics),
                topic_probs=uniform,
                dominant_topic=0,
                dominant_prob=1.0 / self.config.num_topics,
            )

        # Get BoW
        bow = self.dictionary.doc2bow(tokens)

        # Get topic distribution
        topic_dist = self.model.get_document_topics(bow, minimum_probability=0.0)
        probs = np.zeros(self.config.num_topics)
        for topic_id, prob in topic_dist:
            probs[topic_id] = prob

        # Normalize (should already be normalized)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(self.config.num_topics) / self.config.num_topics

        dominant_topic = int(np.argmax(probs))
        dominant_prob = float(probs[dominant_topic])

        return TopicDistribution(
            topic_ids=np.arange(self.config.num_topics),
            topic_probs=probs,
            dominant_topic=dominant_topic,
            dominant_prob=dominant_prob,
        )

    def get_topic_distributions_batch(self, texts: List[str]) -> List[TopicDistribution]:
        """Get topic distributions for multiple texts."""
        return [self.get_topic_distribution(text) for text in texts]

    def get_health_relevance(self, text: str) -> np.ndarray:
        """Get health topic relevance scores R(v_i) for text."""
        if self.health_identifier is None:
            raise RuntimeError("Health topics not identified. Train model first.")

        topic_dist = self.get_topic_distribution(text)
        return self.health_identifier.get_topic_relevance(topic_dist.topic_probs)

    def get_topic_words(self, topic_id: int, topn: int = 10) -> List[Tuple[str, float]]:
        """Get top words for a topic."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.show_topic(topic_id, topn=topn)

    def save(self, path: Optional[str] = None) -> None:
        """Save model to disk."""
        path = Path(path or self.config.model_path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save(str(path / "lda_model"))
        if self.dictionary:
            self.dictionary.save(str(path / "dictionary"))

        # Save config and health topics
        config_data = {
            "num_topics": self.config.num_topics,
            "health_topic_ids": self.health_identifier.health_topic_ids if self.health_identifier else [],
            "health_topic_boost": self.config.health_topic_boost,
            "health_topic_keywords": self.config.health_topic_keywords,
            "trained_at": time.time(),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"LDA model saved to {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load model from disk."""
        path = Path(path or self.config.model_path)

        model_file = path / "lda_model"
        dict_file = path / "dictionary"
        config_file = path / "config.json"

        if not (model_file.exists() and dict_file.exists()):
            logger.warning(f"Model files not found at {path}")
            return False

        try:
            self.model = LdaModel.load(str(model_file))
            self.dictionary = corpora.Dictionary.load(str(dict_file))

            if config_file.exists():
                with open(config_file) as f:
                    config_data = json.load(f)
                self.health_identifier = HealthTopicIdentifier(
                    config_data.get("health_topic_keywords", self.config.health_topic_keywords),
                    config_data.get("health_topic_boost", self.config.health_topic_boost)
                )
                self.health_identifier._health_topic_ids = config_data.get("health_topic_ids", [])

            logger.info(f"LDA model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LDA model: {e}")
            return False

    def print_topics(self, num_topics: int = -1, num_words: int = 10) -> None:
        """Print topics for inspection."""
        if self.model is None:
            logger.warning("Model not trained.")
            return

        topics = self.model.print_topics(num_topics=num_topics, num_words=num_words)
        for topic_id, topic_str in topics:
            health_marker = " [HEALTH]" if self.health_identifier and topic_id in self.health_identifier.health_topic_ids else ""
            print(f"Topic {topic_id}{health_marker}: {topic_str}")


def create_health_corpus(output_path: str, num_abstracts: int = 10000) -> List[str]:
    """
    Create health corpus from PubMed abstracts.
    In practice, download from PubMed API or use existing dataset.
    This creates a placeholder with sample health texts.
    """
    import requests

    logger.info(f"Creating health corpus at {output_path}...")

    # Sample health texts for initial testing
    # In production, fetch from PubMed, Wikipedia health articles, CDC, WHO
    sample_texts = [
        "COVID-19 vaccines are effective at preventing severe disease and hospitalization.",
        "The SARS-CoV-2 virus causes COVID-19 disease which spreads through respiratory droplets.",
        "mRNA vaccines work by teaching cells to make a protein that triggers immune response.",
        "Clinical trials show COVID-19 vaccines have high efficacy against symptomatic infection.",
        "The CDC recommends COVID-19 vaccination for everyone aged 6 months and older.",
        "Vaccine side effects are typically mild including sore arm, fatigue, and fever.",
        "Long COVID can cause persistent symptoms including fatigue, brain fog, and shortness of breath.",
        "New variants of SARS-CoV-2 continue to emerge requiring updated vaccines.",
        "Public health measures like masking and distancing reduce virus transmission.",
        "The WHO declared COVID-19 a pandemic in March 2020.",
        "Influenza vaccines are updated annually to match circulating strains.",
        "Flu vaccination reduces risk of flu illness by 40-60 percent.",
        "Pneumococcal vaccines prevent pneumonia, meningitis, and bloodstream infections.",
        "HPV vaccine prevents cancers caused by human papillomavirus infection.",
        "Measles vaccine is highly effective with two doses providing 97% protection.",
        "Polio has been eliminated in most countries through vaccination programs.",
        "Hepatitis B vaccine prevents liver disease and liver cancer.",
        "Rotavirus vaccine protects children from severe diarrheal disease.",
        "Shingles vaccine reduces risk of shingles and post-herpetic neuralgia.",
        "Tetanus vaccine prevents lockjaw disease from bacterial infection.",
        "Antibiotics treat bacterial infections but are ineffective against viruses.",
        "Antiviral medications can reduce severity of viral infections like influenza.",
        "Monoclonal antibodies provide targeted treatment for COVID-19.",
        "Corticosteroids reduce inflammation in severe COVID-19 cases.",
        "Remdesivir is an antiviral drug approved for COVID-19 treatment.",
        "Paxlovid is an oral antiviral for treating mild to moderate COVID-19.",
        "Ventilators support breathing in critically ill COVID-19 patients.",
        "Clinical trials test safety and efficacy of new treatments.",
        "Randomized controlled trials provide highest quality evidence.",
        "Peer review ensures scientific rigor in medical research.",
        "Meta-analyses combine results from multiple studies.",
        "Systematic reviews synthesize all available evidence on a topic.",
        "The FDA approves drugs based on clinical trial data.",
        "Drug development takes 10-15 years from discovery to approval.",
        "Phase 1 trials test safety in small groups of healthy volunteers.",
        "Phase 2 trials test efficacy and side effects in patients.",
        "Phase 3 trials confirm efficacy in large patient populations.",
        "Phase 4 surveillance monitors drugs after approval.",
        "Vaccine development follows similar phased approach.",
        "Emergency use authorization allows faster access during crises.",
        "Herd immunity occurs when enough people are immune to stop spread.",
        "Vaccine hesitancy is a major challenge for public health.",
        "Misinformation about vaccines spreads rapidly on social media.",
        "Fact-checking helps combat health misinformation.",
        "Health literacy enables people to make informed decisions.",
        "Public trust in health institutions is crucial for compliance.",
        "Epidemiologists study disease patterns in populations.",
        "Contact tracing identifies people exposed to infectious diseases.",
        "Quarantine separates exposed people to prevent spread.",
        "Isolation separates sick people from healthy people.",
        "Testing identifies infected individuals for isolation.",
        "PCR tests detect viral genetic material with high accuracy.",
        "Rapid antigen tests provide quick results at lower sensitivity.",
        "Wastewater surveillance tracks virus levels in communities.",
        "Genomic sequencing identifies new virus variants.",
        "Spike protein mutations affect virus transmissibility and immune escape.",
        "Variant classification includes variants of concern and interest.",
        "Delta variant was more transmissible than original strain.",
        "Omicron variant has many spike protein mutations.",
        "Bivalent vaccines target original strain and Omicron variants.",
        "Vaccine effectiveness varies by variant and time since vaccination.",
        "Booster doses restore protection against new variants.",
        "Immunocompromised people may need additional vaccine doses.",
        "Pregnant women are at higher risk for severe COVID-19.",
        "COVID-19 vaccines are safe during pregnancy.",
        "Children generally have milder COVID-19 than adults.",
        "Multisystem inflammatory syndrome is rare but serious in children.",
        "Schools can operate safely with layered prevention strategies.",
        "Mask mandates in schools reduce COVID-19 transmission.",
        "Ventilation improvements reduce airborne virus spread.",
        "Hand hygiene and surface cleaning reduce transmission risk.",
        "Mental health impacts of pandemic are significant.",
        "Social isolation increases depression and anxiety.",
        "Telehealth expands access to healthcare during pandemic.",
        "Health disparities widened during COVID-19 pandemic.",
        "Essential workers faced higher infection risk.",
        "Vaccine equity is a global health priority.",
        "COVAX facility aims for equitable vaccine distribution.",
        "Intellectual property waivers could increase vaccine production.",
        "mRNA technology enables rapid vaccine development.",
        "mRNA vaccines do not alter human DNA.",
        "Lipid nanoparticles deliver mRNA into cells.",
        "mRNA is degraded by cells after protein production.",
        "No live virus in mRNA vaccines.",
        "Vaccine ingredients are publicly disclosed.",
        "Preservatives like thimerosal removed from most vaccines.",
        "Aluminum adjuvants enhance immune response safely.",
        "Vaccine safety monitoring continues after deployment.",
        "VAERS collects reports of adverse events after vaccination.",
        "V-safe smartphone tool monitors vaccine safety.",
        "Causality assessment distinguishes vaccine effects from coincidence.",
        "Benefits of COVID-19 vaccination far outweigh risks.",
        "Myocarditis risk higher from COVID-19 than vaccination.",
        "Blood clot risk higher from COVID-19 than vaccination.",
        "Guillain-Barré syndrome rare after vaccination.",
        "Anaphylaxis after vaccination is very rare and treatable.",
        "Vaccines do not cause infertility.",
        "Vaccines do not contain microchips or tracking devices.",
        "Vaccines do not shed or transmit to others.",
        "Natural immunity from infection less reliable than vaccination.",
        "Hybrid immunity from infection plus vaccination is strong.",
        "Immunity wanes over time requiring boosters.",
        "T cells provide long-term protection against severe disease.",
        "Antibody levels correlate with protection but not perfectly.",
        "Mucosal immunity may prevent infection better than systemic.",
        "Nasal vaccines in development for mucosal immunity.",
        "Universal coronavirus vaccine research underway.",
        "Pandemic preparedness requires sustained investment.",
        "One Health approach links human, animal, environmental health.",
        "Zoonotic diseases originate from animal reservoirs.",
        "Deforestation increases human-wildlife contact and spillover risk.",
        "Climate change affects infectious disease patterns.",
        "Antimicrobial resistance is a growing global threat.",
        "Antibiotic stewardship preserves effectiveness of antibiotics.",
        "New antibiotics needed to combat resistant bacteria.",
        "Phage therapy as alternative to antibiotics.",
        "Vaccines against bacterial pathogens reduce antibiotic use.",
        "Diabetes increases risk of severe COVID-19.",
        "Obesity is risk factor for severe COVID-19.",
        "Cardiovascular disease worsens COVID-19 outcomes.",
        "Cancer patients may have weaker immune response to vaccines.",
        "Immunosuppressed patients need tailored vaccination schedules.",
        "Autoimmune diseases not caused by vaccines.",
        "Multiple sclerosis patients can safely receive COVID vaccines.",
        "Allergies to vaccine components are rare.",
        "Polyethylene glycol allergy may contraindicate mRNA vaccines.",
        "Latex allergy not a concern for vaccine vial stoppers.",
        "Egg allergy not a contraindication for COVID vaccines.",
        "Prior anaphylaxis to vaccines requires allergist consultation.",
        "Vaccination during breastfeeding is safe.",
        "Vaccine antibodies transfer to infant through breast milk.",
        "No evidence vaccines affect menstrual cycles long-term.",
        "Temporary menstrual changes reported but resolve quickly.",
        "Vaccines do not cause autoimmune diseases.",
        "Vaccines do not cause autism.",
        "Wakefield study linking MMR to autism was fraudulent.",
        "Numerous studies confirm no link between vaccines and autism.",
        "Aluminum in vaccines is less than in diet.",
        "Formaldehyde in vaccines is less than produced by body.",
        "Vaccine schedule is safe and evidence-based.",
        "Spacing out vaccines increases disease risk without benefit.",
        "Combination vaccines reduce number of injections.",
        "Vaccine court compensates rare vaccine injuries.",
        "Vaccine manufacturers have liability protection for public health.",
        "Global vaccine market incentivizes innovation.",
        "GAVI improves vaccine access in low-income countries.",
        "Polio eradication near completion.",
        "Measles elimination achieved in some regions.",
        "Rubella elimination prevents congenital rubella syndrome.",
        "Maternal immunization protects newborns.",
        "Pertussis vaccination during pregnancy protects infants.",
        "Influenza vaccination during pregnancy protects mother and baby.",
        "RSV vaccine for pregnant women protects infants.",
        "Group B strep vaccine in development for pregnant women.",
        "Cytomegalovirus vaccine could prevent birth defects.",
        "Zika vaccine development accelerated after 2015 outbreak.",
        "Ebola vaccine deployed in outbreaks.",
        "Malaria vaccine RTS,S reduces severe malaria in children.",
        "Tuberculosis vaccine BCG used for 100 years.",
        "New TB vaccines in clinical trials.",
        "HIV vaccine remains elusive after decades.",
        "Broadly neutralizing antibodies guide HIV vaccine design.",
        "mRNA technology applied to HIV vaccine development.",
        "Therapeutic vaccines for chronic infections.",
        "Cancer vaccines target tumor-specific antigens.",
        "HPV vaccine prevents cervical, anal, throat cancers.",
        "Hepatitis B vaccine prevents liver cancer.",
        "Personalized neoantigen vaccines for cancer treatment.",
        "Prostate cancer vaccine sipuleucel-T approved.",
        "Melanoma vaccine shows promise in trials.",
        "Vaccine adjuvants enhance immune response.",
        "Alum adjuvant used for decades.",
        "MF59 adjuvant in flu vaccines for elderly.",
        "AS01 adjuvant in shingles and malaria vaccines.",
        "TLR agonists as novel adjuvants.",
        "Nanoparticle vaccines display antigens effectively.",
        "Self-amplifying mRNA vaccines need lower doses.",
        "DNA vaccines stable at room temperature.",
        "Viral vector vaccines use harmless viruses.",
        "Adenovirus vectors in COVID vaccines.",
        "AAV vectors for gene therapy.",
        "Protein subunit vaccines are safe and stable.",
        "Novavax COVID vaccine is protein subunit.",
        "VLP vaccines mimic virus structure.",
        "HPV vaccine is VLP-based.",
        "Hepatitis B vaccine is recombinant protein.",
        "Conjugate vaccines link polysaccharide to protein.",
        "Pneumococcal conjugate vaccine covers multiple serotypes.",
        "MenACWY vaccine prevents meningococcal disease.",
        "Hib vaccine eliminated Haemophilus influenzae type b.",
        "Typhoid conjugate vaccine for endemic areas.",
        "Cholera vaccine for outbreak response.",
        "Rotavirus vaccine live attenuated.",
        "Oral polio vaccine live attenuated.",
        "Measles vaccine live attenuated.",
        "Varicella vaccine live attenuated.",
        "Yellow fever vaccine live attenuated.",
        "Live vaccines contraindicated in immunocompromised.",
        "Inactivated vaccines safe for all.",
        "Toxoid vaccines target bacterial toxins.",
        "Tetanus toxoid in DTaP/Tdap.",
        "Diphtheria toxoid in DTaP/Tdap.",
        "Pertussis component acellular in DTaP.",
        "Whole cell pertussis more reactogenic.",
        "Vaccine schedules vary by country.",
        "WHO recommends core vaccines for all children.",
        "BCG at birth in TB-endemic countries.",
        "Hepatitis B at birth globally.",
        "Polio multiple doses in first year.",
        "DTP at 6, 10, 14 weeks.",
        "Measles at 9-12 months.",
        "Second measles dose at 15-18 months.",
        "Booster doses throughout childhood.",
        "Adolescent vaccines include HPV, MenACWY, Tdap.",
        "Adult boosters for tetanus, diphtheria, pertussis.",
        "Annual flu vaccine for all ages.",
        "Shingles vaccine at 50+.",
        "Pneumococcal vaccine at 65+.",
        "COVID vaccines updated for variants.",
        "Travel vaccines for endemic areas.",
        "Yellow fever for Africa, South America.",
        "Japanese encephalitis for Asia.",
        "Typhoid for South Asia.",
        "Rabies pre-exposure for high risk.",
        "Hepatitis A for travelers.",
        "Meningococcal for Hajj pilgrims.",
        "Polio booster for endemic areas.",
        "Cholera for humanitarian crises.",
        "Ebola for outbreak responders.",
        "Smallpox eradicated 1980.",
        "Last natural smallpox case 1977.",
        "Smallpox vaccine from cowpox virus.",
        "Jenner developed first vaccine 1796.",
        "Pasteur developed rabies vaccine 1885.",
        "Salk inactivated polio vaccine 1955.",
        "Sabin oral polio vaccine 1961.",
        "Measles vaccine 1963.",
        "MMR combined 1971.",
        "Hepatitis B vaccine 1981.",
        "Hib vaccine 1985.",
        "Varicella vaccine 1995.",
        "Rotavirus vaccine 2006.",
        "HPV vaccine 2006.",
        "COVID-19 vaccines 2020.",
        "mRNA vaccines Nobel Prize 2023.",
        "Katalin Karikó and Drew Weissman pioneered mRNA.",
        "Lipid nanoparticles by Pieter Cullis.",
        "Vaccine development accelerated by pandemic.",
        "Regulatory flexibility maintained safety standards.",
        "Real-world evidence confirms trial results.",
        "Vaccines saved millions of lives.",
        "Vaccines are one of greatest public health achievements.",
    ]

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for text in sample_texts:
            f.write(text + "\n")

    logger.info(f"Created sample corpus with {len(sample_texts)} texts at {output_path}")
    return sample_texts


# Example usage and testing
if __name__ == "__main__":
    # Create sample corpus
    corpus_path = "data/corpus/health_corpus.txt"
    texts = create_health_corpus(corpus_path, num_abstracts=1000)

    # Train LDA
    config = LDAConfig(num_topics=20, model_path="data/cache/lda_model")
    lda = LDATopicModel(config)
    lda.train(texts)

    # Print topics
    lda.print_topics()

    # Test topic distribution
    test_sentence = "COVID-19 vaccines prevent severe disease and hospitalization."
    dist = lda.get_topic_distribution(test_sentence)
    print(f"\nTest sentence: {test_sentence}")
    print(f"Dominant topic: {dist.dominant_topic} (prob: {dist.dominant_prob:.4f})")
    print(f"Health relevance: {lda.get_health_relevance(test_sentence)}")