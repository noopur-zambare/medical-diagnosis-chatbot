import os
import json
import re
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_openai import ChatOpenAI

from vector_store import upsert_report, query_similar
from Agents import (
    Cardiologist,
    Gastroenterologist,
    Pulmonologist,
    Dermatologist,
    Rheumatologist,
    CollectAll,
)

# Setup 
load_dotenv("api.env")
os.makedirs("results", exist_ok=True)

with open("keyword.json", "r") as f:
    KEYWORDS = json.load(f)


def route_agents(user_query: str):
    """Keyword routing (word-boundary match to reduce false positives)."""
    user_query = (user_query or "").lower()
    activated_agents = []
    for specialist, terms in KEYWORDS.items():
        for term in terms:
            term_l = term.lower().strip()
            if not term_l:
                continue
            if re.search(r"\b" + re.escape(term_l) + r"\b", user_query):
                activated_agents.append(specialist)
                break
    return activated_agents


agent_classes = {
    "Cardiologist": Cardiologist,
    "Gastroenterologist": Gastroenterologist,
    "Pulmonologist": Pulmonologist,
    "Dermatologist": Dermatologist,
    "Rheumatologist": Rheumatologist,
}


def _fmt_score(score):
    try:
        return f"{float(score):.4f}"
    except Exception:
        return str(score)


# Pipeline
def run_pipeline(medical_report: str, patient_id: str, top_k: int):
    medical_report = (medical_report or "").strip()
    patient_id = (patient_id or "").strip() or "patient1"
    top_k = int(top_k or 3)

    if not medical_report:
        return (
            "Please enter symptoms/report text.",
            "",
            "None",
            "",
            "",
            {},
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_id = f"{patient_id}_{timestamp}"

    # Retrieve similar reports
    top_results = query_similar(medical_report, patient_id, top_k=top_k) or {}
    matches = top_results.get("matches", []) or []

    retrieval_md = []
    retrieval_texts = []
    for i, match in enumerate(matches, start=1):
        score = match.get("score", "N/A")
        text = (match.get("metadata", {}) or {}).get("report_text", "") or ""
        retrieval_texts.append(text)

        retrieval_md.append(
            f"**Result {i}**\n\n"
            f"- **Score:** `{_fmt_score(score)}`\n"
            f"- **Report:** {text}"
        )

    retrieval_out = "\n\n---\n\n".join(retrieval_md) if retrieval_md else "No matches found."

    # Summarize past history
    if retrieval_texts:
        summarizer = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        summary_prompt = (
            "You are given multiple medical reports for the same patient.\n"
            "Synthesize them into ONE short, well-structured paragraph.\n"
            "Highlight major symptoms, diagnoses, patterns, and clinically significant overlaps.\n"
            "Keep it concise and medically accurate.\n\n"
            + "\n\n".join(retrieval_texts)
        )
        summary_text = summarizer.invoke(summary_prompt).content
    else:
        summary_text = "No prior similar reports found."

    # Route agents
    activated_roles = route_agents(medical_report)
    activated_out = ", ".join(activated_roles) if activated_roles else "None"

    # Run specialist agents
    agents = {
        role: agent_classes[role](medical_report, past_history=summary_text)
        for role in activated_roles
        if role in agent_classes
    }

    specialist_reports = {}
    specialists_md = []
    for role, agent in agents.items():
        try:
            result = agent.run()
        except Exception as e:
            result = f"Error running {role}: {e}"
        specialist_reports[role] = result
        specialists_md.append(f"## {role}\n\n{result}")

    specialists_out = (
        "\n\n---\n\n".join(specialists_md) if specialists_md else "No specialist agents activated."
    )

    # Collect all reports
    try:
        team_agent = CollectAll(
            specialist_reports=specialist_reports,
            past_history=summary_text,
        )
        final_diagnosis_text = team_agent.run()
    except Exception as e:
        final_diagnosis_text = f"Error running CollectAll: {e}"

    # Upsert final result
    try:
        upsert_report(report_id, medical_report, patient_id)
        upsert_status = f"Upserted to vector store as `{report_id}` for `{patient_id}`."
    except Exception as e:
        upsert_status = f"Upsert failed: {e}"

    final_out = final_diagnosis_text

    full_report = {
        "Retrieved Reports": retrieval_out,
        "Summary": summary_text,
        "Activated Agents": activated_out,
        "Specialist Reports": specialists_out,
        "Final Diagnosis": final_out,
    }

    return retrieval_out, summary_text, activated_out, specialists_out, final_out, full_report


# Save report 
def save_report(patient_id, full_report_dict, save_as, custom_name, sections):
    patient_id = (patient_id or "").strip() or "patient1"
    full_report_dict = full_report_dict or {}

    if save_as == "None":
        return "<p style='font-size:14px; color:red;'>Not saved.</p>"

    if not sections:
        return "<p style='font-size:14px; color:red;'>Select at least one section to save.</p>"

    section_map = {
        "Retrieved": "Retrieved Reports",
        "Summary": "Summary",
        "Agents": "Activated Agents",
        "Specialists": "Specialist Reports",
        "Final": "Final Diagnosis",
    }

    content = ""
    for section in sections:
        key = section_map.get(section, section)
        if key in full_report_dict:
            content += f"=== {key} ===\n{full_report_dict[key]}\n\n"

    if not content.strip():
        return "<p style='font-size:14px; color:red;'>Nothing to save yet. Run the pipeline first.</p>"

    if (custom_name or "").strip():
        filename = custom_name.strip()
    else:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_{date_str}"

    if save_as == "txt":
        filepath = os.path.join("results", filename + ".txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    elif save_as == "pdf":
        filepath = os.path.join("results", filename + ".pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in content.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.output(filepath)

    else:
        return "<p style='font-size:14px; color:red;'>Invalid save type.</p>"

    return f"<p style='font-size:14px; color:green;'>Saved as <b>{filepath}</b></p>"


def collect_sections(r, s, a, sp, f):
    selected = []
    if r:
        selected.append("Retrieved")
    if s:
        selected.append("Summary")
    if a:
        selected.append("Agents")
    if sp:
        selected.append("Specialists")
    if f:
        selected.append("Final")
    return selected


# Gradio UI
with gr.Blocks(title="Multi-Agent Medical Diagnosis Chatbot") as demo:
    gr.Markdown("<h1 style='text-align:center; color:orange;'>Multi-Agent Medical Diagnosis Chatbot</h1>")

    with gr.Row():
        medical_report_in = gr.Textbox(
            label="Symptoms",
            placeholder="Type symptoms here (e.g., knee joint pain, stomach pain)...",
            lines=4,
        )

    with gr.Row():
        patient_id_in = gr.Textbox(label="Patient ID", value="patient1")
        top_k_in = gr.Slider(label="Top-K similar reports", minimum=1, maximum=10, step=1, value=3)

    run_btn = gr.Button("Run")
    gr.Markdown("<hr style='border:0; border-top:0.5px solid white;'>")
    gr.Markdown("<h2 style='color:orange;'>Retrieved Similar Reports</h2>")
    retrieval_out = gr.Markdown(value="")

    gr.Markdown("<hr style='border:0; border-top:0.5px solid white;'>")
    gr.Markdown("<h2 style='color:orange;'>Summarized Past Medical History</h2>")
    summary_out = gr.Markdown(value="")

    gr.Markdown("<hr style='border:0; border-top:0.5px solid white;'>")
    gr.Markdown("<h2 style='color:orange;'>Active Agents</h2>")
    activated_out = gr.Textbox(label="Currently active agents based on user",interactive=False)

    gr.Markdown("<hr style='border:0; border-top:0.5px solid white;'>")
    gr.Markdown("<h2 style='color:orange;'>Specialist Reports</h2>")
    specialists_out = gr.Markdown(value="")

    gr.Markdown("<hr style='border:0; border-top:0.5px solid white;'>")
    gr.Markdown("<h2 style='color:orange;'>Final Diagnosis</h2>")
    final_out = gr.Markdown(value="")
    gr.Markdown("<hr style='border:0; border-top:0.5px solid white;'>")

    full_report_state = gr.State({})

    run_btn.click(
        fn=run_pipeline,
        inputs=[medical_report_in, patient_id_in, top_k_in],
        outputs=[retrieval_out, summary_out, activated_out, specialists_out, final_out, full_report_state],
    )

    gr.Markdown("<h2 style='color:orange;'>Save Report</h2>")


    with gr.Row(elem_id="save-row"):
        with gr.Column(scale=1, elem_id="col-filetype"):
            save_as_option = gr.Dropdown(
                choices=["None", "txt", "pdf"],
                value="None",
                label="File type",
                interactive=True,
                allow_custom_value=False
            )
        with gr.Column(scale=2, elem_id="col-filename"):
            custom_name_in = gr.Textbox(
                label="Filename (optional)",
                placeholder="automatically set, if blank",
                lines=1,
            )

        with gr.Column(scale=3, elem_id="col-checkboxes"):
            gr.Markdown("**Select Sections to Save:**")
            with gr.Group():
                with gr.Row():
                    cb_retrieved = gr.Checkbox(label="Retrieved", value=False)
                    cb_summary = gr.Checkbox(label="Summary", value=False)
                    cb_agents = gr.Checkbox(label="Agents", value=False)
                    cb_specialists = gr.Checkbox(label="Specialists", value=False)
                    cb_final = gr.Checkbox(label="Final", value=True)

                   
    with gr.Row(elem_id="save-row"):
        save_btn = gr.Button("Save", elem_classes="white-btn")

    save_out = gr.Markdown()

    save_btn.click(
        fn=lambda patient_id, state, save_as, name, r, s, a, sp, f: save_report(
            patient_id,
            state,
            save_as,
            name,
            collect_sections(r, s, a, sp, f),
        ),
        inputs=[
            patient_id_in,
            full_report_state,
            save_as_option,
            custom_name_in,
            cb_retrieved,
            cb_summary,
            cb_agents,
            cb_specialists,
            cb_final,
        ],
        outputs=[save_out],
    )

    demo.css = """
    .gr-markdown { font-size:14px !important; }
    #save-row { justify-content: center; margin-top:12px; }
    .white-btn {
        background-color:white !important;
        color:black !important;
        border:1px solid orange;
        padding:8px 18px;
    }

    """

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
