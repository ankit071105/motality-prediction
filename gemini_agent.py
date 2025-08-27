import google.generativeai as genai
from decouple import config
import pandas as pd
import json
import re

class GeminiMortalityAgent:
    def __init__(self):
        # Configure Gemini API (you'll need to set up your API key)
        try:
            api_key = config('GEMINI_API_KEY')
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.available = True
        except:
            print("Gemini API not configured. Using simulated responses.")
            self.available = False
    
    def analyze_risk_factors(self, patient_data):
        """Use Gemini to analyze risk factors and provide explanations"""
        prompt = f"""
        Analyze this patient's health data and provide insights on mortality risk factors:
        
        {patient_data.to_dict()}
        
        Please provide:
        1. A summary of the key risk factors
        2. Explanation of how each factor contributes to mortality risk
        3. Recommendations for risk mitigation
        4. Estimated impact of modifying each risk factor
        
        Format your response as JSON with keys: summary, risk_factors, recommendations, impact_analysis.
        """
        
        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return self._parse_gemini_response(response.text)
            except Exception as e:
                print(f"Gemini API error: {e}")
                return self._get_simulated_response(patient_data)
        else:
            return self._get_simulated_response(patient_data)
    
    def predict_time_to_event(self, patient_data, risk_score):
        """Use Gemini to provide a natural language prediction of time to event"""
        prompt = f"""
        Based on this patient's health profile and a calculated risk score of {risk_score:.2f}, 
        provide a prediction of time to mortality event:
        
        {patient_data.to_dict()}
        
        Consider factors like age, comorbidities, and lifestyle factors.
        Provide a realistic time estimate and confidence level.
        
        Format your response as JSON with keys: time_estimate, confidence, explanation.
        """
        
        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return self._parse_time_response(response.text)
            except Exception as e:
                print(f"Gemini API error: {e}")
                return self._get_simulated_time_response(risk_score)
        else:
            return self._get_simulated_time_response(risk_score)
    
    def _parse_gemini_response(self, response_text):
        """Parse Gemini's response into structured data"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_simulated_response()
        except:
            return self._get_simulated_response()
    
    def _parse_time_response(self, response_text):
        """Parse time prediction response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_simulated_time_response(0.5)
        except:
            return self._get_simulated_time_response(0.5)
    
    def _get_simulated_response(self, patient_data=None):
        """Fallback simulated response when Gemini is not available"""
        return {
            "summary": "Based on the health data provided, several risk factors contribute to mortality risk.",
            "risk_factors": [
                {"factor": "Age", "impact": "High", "explanation": "Advanced age is a significant risk factor for mortality."},
                {"factor": "Blood Pressure", "impact": "Medium", "explanation": "Elevated blood pressure increases cardiovascular risk."}
            ],
            "recommendations": [
                "Regular health screenings",
                "Lifestyle modifications including diet and exercise",
                "Medication adherence if prescribed"
            ],
            "impact_analysis": "Reducing blood pressure by 10mmHg could decrease mortality risk by approximately 15%."
        }
    
    def _get_simulated_time_response(self, risk_score):
        """Fallback simulated time prediction"""
        if risk_score > 0.7:
            estimate = "1-3 years"
            confidence = "High"
        elif risk_score > 0.4:
            estimate = "3-8 years"
            confidence = "Medium"
        else:
            estimate = "8+ years"
            confidence = "Low"
        
        return {
            "time_estimate": estimate,
            "confidence": confidence,
            "explanation": f"Based on a risk score of {risk_score:.2f}, the estimated time to mortality event is {estimate} with {confidence.lower()} confidence."
        }

# Create a global instance
gemini_agent = GeminiMortalityAgent()