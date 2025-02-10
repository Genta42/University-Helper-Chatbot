import logging
from bert_score import score as bert_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import requests
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to evaluate BERTScore
def evaluate_bertscore(references, candidates, lang="en"):
    logging.debug("Evaluating BERTScore...")

    # Ensure valid inputs
    references = [ref if ref.strip() else "No reference available." for ref in references]
    candidates = [cand if cand.strip() else "No candidate response." for cand in candidates]

    logging.debug(f"References: {references}")
    logging.debug(f"Candidates: {candidates}")

    try:
        # Compute BERTScore with baseline correction
        P, R, F1 = bert_score(candidates, references, lang=lang, rescale_with_baseline=True)

        # Ensure scores are non-negative
        avg_precision = max(P.mean().item(), 0)
        avg_recall = max(R.mean().item(), 0)
        avg_f1 = max(F1.mean().item(), 0)

        logging.debug(f"BERTScore - Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}")
        return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}

    except Exception as e:
        logging.error(f"Error calculating BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

# Function to evaluate Perplexity
def evaluate_perplexity(candidates, model_name="gpt2"):
    logging.debug("Evaluating Perplexity...")

    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()

        perplexities = []
        for candidate in candidates:
            if not candidate.strip():  # Skip empty responses
                logging.warning("Skipping empty candidate response.")
                continue

            inputs = tokenizer(candidate, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                log_likelihood = outputs.loss.item()
                
                # Normalize Perplexity by sentence length
                sentence_length = inputs["input_ids"].shape[1]
                perplexity = torch.exp(torch.tensor(log_likelihood / sentence_length)).item()
                
                perplexities.append(perplexity)

        avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("nan")

        logging.debug(f"Average Perplexity: {avg_perplexity}")
        return avg_perplexity

    except Exception as e:
        logging.error(f"Error calculating Perplexity: {e}")
        return float("nan")

# Fetch responses from the chatbot
def fetch_responses_from_chatbot(queries, url="http://localhost:3000/chat"):
    logging.debug(f"Fetching responses from chatbot at {url}")
    responses = []

    for query in queries:
        try:
            response = requests.post(url, json={"userInput": query})
            if response.status_code == 200:
                bot_response = response.json().get("response", "").strip()
                logging.debug(f"Query: {query}, Response: {bot_response}")
                responses.append(bot_response if bot_response else "No response.")
            else:
                logging.warning(f"Failed to fetch response for query '{query}': HTTP {response.status_code}")
                responses.append("No response.")

        except requests.RequestException as e:
            logging.error(f"Error fetching response for query '{query}': {e}")
            responses.append("No response.")

    return responses

# Main script for testing
if __name__ == "__main__":
    # Define test queries and expected references
    test_queries = [
        "Where can I find tutoring support for my courses?",
        "How do I contact the financial aid office?",
        "What resources are available for mental health?"
    ]

    reference_responses = [
        "You can find tutoring support at the academic resource center.",
        "You can contact the financial aid office via email at finaid@university.edu.",
        "Mental health resources are available at the counseling center."
    ]

    # Fetch responses from the chatbot
    candidate_responses = fetch_responses_from_chatbot(test_queries)

    # Evaluate BERTScore
    try:
        bertscore_results = evaluate_bertscore(reference_responses, candidate_responses)
        print(f"\n🔹 BERTScore Results:")
        print(f"📌 Precision: {bertscore_results['precision']:.4f}")
        print(f"📌 Recall: {bertscore_results['recall']:.4f}")
        print(f"📌 F1 Score: {bertscore_results['f1']:.4f}")
    except Exception as e:
        logging.error(f"An error occurred during BERTScore evaluation: {e}")

    # Evaluate Perplexity
    try:
        avg_perplexity = evaluate_perplexity(candidate_responses)
        print(f"\n🔹 Perplexity Results:")
        print(f"📌 Average Perplexity: {avg_perplexity:.4f}")
    except Exception as e:
        logging.error(f"An error occurred during Perplexity evaluation: {e}")