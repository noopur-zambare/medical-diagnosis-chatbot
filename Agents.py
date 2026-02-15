from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Agent:
    def __init__(self, medical_report=None, role=None, past_history=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.past_history = past_history 
        self.extra_info = extra_info or {}
        self.prompt_template = self.create_prompt_template()
        self.model = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"
        )

    def create_prompt_template(self):
        history_text = f"Past Patient History: {self.past_history}\n\n" if self.past_history else ""

        if self.role == "CollectAll":

          active_reports = {
              key.replace("_report", "").capitalize(): value
              for key, value in self.extra_info.items()
              if value and value.strip() != ""
          }

          reports_text = ""
          for role, report in active_reports.items():
              reports_text += f"\n{role} Report:\n{report}\n"

          active_roles_text = ", ".join(active_reports.keys()) if active_reports else "No active specialists"

          templates = f"""
              Act as a multidisciplinary healthcare team consisting ONLY of the following active specialists:
              {active_roles_text}.

              Task:
              Review ONLY the provided specialist reports.
              Do NOT assume involvement of any specialist whose report is missing.
              Provide one concise paragraph summarizing up to 3 possible health issues and the reasoning based strictly on the available reports.

              Active Specialist Reports:
              {reports_text}
          """

        else:
            templates = {
                "Cardiologist": f"""
                    Act like a cardiologist.
                    {history_text}
                    Task: Review the patient's cardiac workup and provide possible causes of symptoms and recommended next steps.
                    Medical Report: {{medical_report}}
                """,
                "Gastroenterologist": f"""
                    Act like a gastroenterologist.
                    {history_text}
                    Task: Review the patient's report and provide gastroenterological assessment and recommended next steps.
                    Patient's Report: {{medical_report}}
                """,
                "Pulmonologist": f"""
                    Act like a pulmonologist.
                    {history_text}
                    Task: Review the patient's report and provide pulmonary assessment and recommended next steps.
                    Patient's Report: {{medical_report}}
                """,
                "Dermatologist": f"""
                    Act like a dermatologist.
                    {history_text}
                    Task: Review the patient's report and provide dermatological assessment and recommended next steps.
                    Patient's Report: {{medical_report}}
                """,
                "Rheumatologist": f"""
                    Act like a rheumatologist.
                    {history_text}
                    Task: Review the patient's report and provide rheumatological assessment and recommended next steps.
                    Patient's Report: {{medical_report}}
                """
            }

            templates = templates[self.role]

        return PromptTemplate.from_template(templates)

    def run(self):
        print(f"{self.role} is running...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print("Error occurred:", e)
            return None


class Cardiologist(Agent):
    def __init__(self, medical_report, past_history=None):
        super().__init__(medical_report, "Cardiologist", past_history=past_history)


class Gastroenterologist(Agent):
    def __init__(self, medical_report, past_history=None):
        super().__init__(medical_report, "Gastroenterologist", past_history=past_history)


class Pulmonologist(Agent):
    def __init__(self, medical_report, past_history=None):
        super().__init__(medical_report, "Pulmonologist", past_history=past_history)


class Dermatologist(Agent):
    def __init__(self, medical_report, past_history=None):
        super().__init__(medical_report, "Dermatologist", past_history=past_history)


class Rheumatologist(Agent):
    def __init__(self, medical_report, past_history=None):
        super().__init__(medical_report, "Rheumatologist", past_history=past_history)


class CollectAll(Agent):
    def __init__(self,
                 specialist_reports: dict,
                 past_history=None):

        super().__init__(role="CollectAll",
                         extra_info=specialist_reports,
                         past_history=past_history)

