import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic health data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic features
        age = np.random.normal(65, 15, n_samples)
        age = np.clip(age, 40, 100)  # Clip age between 40-100
        
        blood_pressure = np.random.normal(120, 20, n_samples)
        cholesterol = np.random.normal(200, 40, n_samples)
        bmi = np.random.normal(28, 5, n_samples)
        glucose = np.random.normal(100, 20, n_samples)
        
        # Generate binary features
        smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        diabetes = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        family_history = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        physical_activity = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        
        # Generate more detailed health features
        heart_rate = np.random.normal(75, 10, n_samples)
        respiratory_rate = np.random.normal(16, 3, n_samples)
        oxygen_saturation = np.random.normal(98, 2, n_samples)
        creatinine = np.random.normal(1.0, 0.3, n_samples)
        
        # Generate categorical features
        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
        ethnicities = np.random.choice(['Caucasian', 'African', 'Asian', 'Hispanic'], n_samples)
        blood_types = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_samples)
        
        # Generate target variable with some relationship to features
        # Higher risk with age, smoking, diabetes, high BP, high cholesterol
        risk_score = (
            (age/10) + 
            smoking*2 + 
            diabetes*3 + 
            (blood_pressure-120)/10 + 
            (cholesterol-200)/50 +
            (bmi-25)/5 +
            (glucose-100)/20 +
            (100 - oxygen_saturation) * 0.5
        )
        
        # Add some randomness
        risk_score += np.random.normal(0, 1, n_samples)
        
        # Create time-to-event data (in months)
        base_survival = 120  # 10 years base survival
        time_to_event = base_survival / (1 + np.exp(risk_score/10))
        
        # Create event indicator (1 if death occurred within observation period)
        event_indicator = (time_to_event < 60).astype(int)  # 5-year observation period
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': genders,
            'ethnicity': ethnicities,
            'blood_type': blood_types,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'bmi': bmi,
            'glucose': glucose,
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'oxygen_saturation': oxygen_saturation,
            'creatinine': creatinine,
            'smoking': smoking,
            'diabetes': diabetes,
            'family_history': family_history,
            'physical_activity': physical_activity,
            'time_to_event': time_to_event,
            'event': event_indicator
        })
        
        return data
    
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'ethnicity', 'blood_type']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Separate features and target
        X = df_processed.drop(['time_to_event', 'event'], axis=1)
        y = df_processed['event']
        
        # Scale numerical features - fit the scaler on all features at once
        numerical_cols = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'glucose', 
                         'heart_rate', 'respiratory_rate', 'oxygen_saturation', 'creatinine']
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def preprocess_new_data(self, df):
        """Preprocess new data for prediction using fitted encoders and scaler"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'ethnicity', 'blood_type']
        for col in categorical_cols:
            if col in df_processed.columns and col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Ensure all expected columns are present
        expected_cols = ['age', 'gender', 'ethnicity', 'blood_type', 'blood_pressure', 
                        'cholesterol', 'bmi', 'glucose', 'heart_rate', 'respiratory_rate', 
                        'oxygen_saturation', 'creatinine', 'smoking', 'diabetes', 
                        'family_history', 'physical_activity']
        
        for col in expected_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0  # Fill missing columns with 0
        
        # Reorder columns to match training data
        df_processed = df_processed[expected_cols]
        
        # Scale numerical features - transform all at once
        numerical_cols = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'glucose', 
                         'heart_rate', 'respiratory_rate', 'oxygen_saturation', 'creatinine']
        
        # Scale all numerical features together
        df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.generate_synthetic_data()
    print("Synthetic data generated with shape:", data.shape)
    print("Mortality distribution:\n", data['event'].value_counts())
    data.to_csv('synthetic_health_data.csv', index=False)