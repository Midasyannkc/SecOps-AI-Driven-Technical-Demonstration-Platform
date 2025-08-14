#!/usr/bin/env python3
"""
AI-Driven Threat Detection Engine
Core machine learning models and real-time processing for cybersecurity threat identification
"""

import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import json
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from collections import defaultdict, deque
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThreatEvent:
    """Structured representation of a security event"""
    timestamp: datetime
    source_ip: str
    destination_ip: str
    event_type: str
    severity: int
    raw_data: Dict
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    process_name: Optional[str] = None
    command_line: Optional[str] = None
    network_protocol: Optional[str] = None
    bytes_transferred: Optional[int] = None

@dataclass
class ThreatAlert:
    """AI-generated threat alert with risk assessment"""
    alert_id: str
    threat_type: str
    confidence_score: float
    risk_score: int  # 1-100 scale
    affected_assets: List[str]
    attack_vector: str
    business_impact: str
    recommended_actions: List[str]
    detection_time: datetime
    evidence: List[ThreatEvent]
    
class AIThreatDetector:
    """
    Advanced AI-driven threat detection system combining multiple ML approaches
    """
    
    def __init__(self, model_path: str = "ai_engine/ml_models/"):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.threat_patterns = {}
        self.behavioral_baselines = defaultdict(dict)
        self.event_buffer = deque(maxlen=10000)
        self.active_investigations = {}
        
        # Load pre-trained models
        self._load_models()
        
        # Initialize real-time processing
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        
    def _load_models(self):
        """Load pre-trained ML models and preprocessing components"""
        try:
            # Anomaly detection model for behavioral analysis
            self.models['anomaly_detector'] = joblib.load(f"{self.model_path}anomaly_detector.pkl")
            self.scalers['anomaly'] = joblib.load(f"{self.model_path}anomaly_scaler.pkl")
            
            # Threat classification model
            self.models['threat_classifier'] = joblib.load(f"{self.model_path}threat_classifier.pkl")
            self.encoders['threat_features'] = joblib.load(f"{self.model_path}threat_encoder.pkl")
            
            # Risk scoring model
            self.models['risk_scorer'] = joblib.load(f"{self.model_path}risk_scorer.pkl")
            
            # Load deep learning model for advanced pattern recognition
            self.models['deep_analyzer'] = tf.keras.models.load_model(f"{self.model_path}deep_threat_model.h5")
            
            # Load threat intelligence patterns
            with open(f"{self.model_path}threat_patterns.json", 'r') as f:
                self.threat_patterns = json.load(f)
                
            logger.info("All AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Initialize fallback models for demonstration
            self._initialize_demo_models()
    
    def _initialize_demo_models(self):
        """Initialize demonstration models with synthetic training data"""
        logger.info("Initializing demonstration models...")
        
        # Create synthetic training data for demo purposes
        demo_features = np.random.rand(1000, 15)
        demo_labels = np.random.randint(0, 5, 1000)  # 5 threat types
        
        # Train basic models for demonstration
        self.models['anomaly_detector'] = IsolationForest(contamination=0.1, random_state=42)
        self.models['anomaly_detector'].fit(demo_features)
        
        self.models['threat_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['threat_classifier'].fit(demo_features, demo_labels)
        
        self.scalers['anomaly'] = StandardScaler()
        self.scalers['anomaly'].fit(demo_features)
        
        # Demo threat patterns
        self.threat_patterns = {
            "apt_indicators": [
                "powershell.exe -EncodedCommand",
                "cmd.exe /c echo",
                "net user /add",
                "reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
            ],
            "ransomware_indicators": [
                ".locked",
                ".encrypted",
                "Your files have been encrypted",
                "bitcoin payment"
            ],
            "lateral_movement": [
                "psexec",
                "wmic process call create",
                "net use \\\\",
                "winrm"
            ]
        }
        
        logger.info("Demo models initialized successfully")
    
    async def process_event(self, event: ThreatEvent) -> Optional[ThreatAlert]:
        """
        Process a single security event through AI analysis pipeline
        """
        try:
            # Add to event buffer for correlation analysis
            self.event_buffer.append(event)
            
            # Extract features for ML analysis
            features = self._extract_features(event)
            
            # Run through detection pipeline
            anomaly_score = self._detect_anomaly(features)
            threat_classification = self._classify_threat(features, event)
            risk_score = self._calculate_risk_score(features, anomaly_score, threat_classification)
            
            # Behavioral analysis
            behavioral_anomaly = self._analyze_behavioral_patterns(event)
            
            # Correlation with recent events
            correlated_events = self._correlate_events(event)
            
            # Determine if this warrants an alert
            if self._should_generate_alert(anomaly_score, risk_score, behavioral_anomaly):
                alert = await self._generate_threat_alert(
                    event, features, anomaly_score, threat_classification, 
                    risk_score, correlated_events
                )
                return alert
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing event: {str(e)}")
            return None
    
    def _extract_features(self, event: ThreatEvent) -> np.ndarray:
        """Extract numerical features from security event for ML analysis"""
        features = []
        
        # Time-based features
        hour_of_day = event.timestamp.hour
        day_of_week = event.timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_business_hours = 1 if 8 <= hour_of_day <= 18 else 0
        
        features.extend([hour_of_day, day_of_week, is_weekend, is_business_hours])
        
        # Network features
        source_ip_entropy = self._calculate_ip_entropy(event.source_ip) if event.source_ip else 0
        dest_ip_entropy = self._calculate_ip_entropy(event.destination_ip) if event.destination_ip else 0
        bytes_transferred_log = np.log1p(event.bytes_transferred) if event.bytes_transferred else 0
        
        features.extend([source_ip_entropy, dest_ip_entropy, bytes_transferred_log])
        
        # Event type encoding
        event_type_encoded = hash(event.event_type) % 1000 / 1000 if event.event_type else 0
        severity_normalized = event.severity / 10 if event.severity else 0
        
        features.extend([event_type_encoded, severity_normalized])
        
        # Process and command analysis
        process_risk_score = self._calculate_process_risk(event.process_name) if event.process_name else 0
        command_risk_score = self._calculate_command_risk(event.command_line) if event.command_line else 0
        
        features.extend([process_risk_score, command_risk_score])
        
        # User behavior features
        user_risk_score = self._get_user_risk_score(event.user_id) if event.user_id else 0
        device_risk_score = self._get_device_risk_score(event.device_id) if event.device_id else 0
        
        features.extend([user_risk_score, device_risk_score])
        
        # Frequency analysis
        recent_similar_events = self._count_recent_similar_events(event)
        features.append(recent_similar_events)
        
        return np.array(features).reshape(1, -1)
    
    def _detect_anomaly(self, features: np.ndarray) -> float:
        """Use isolation forest to detect anomalous patterns"""
        try:
            # Scale features
            scaled_features = self.scalers['anomaly'].transform(features)
            
            # Get anomaly score (-1 for outliers, 1 for inliers)
            anomaly_prediction = self.models['anomaly_detector'].predict(scaled_features)[0]
            anomaly_score = self.models['anomaly_detector'].decision_function(scaled_features)[0]
            
            # Convert to 0-1 scale where higher values indicate more anomalous
            normalized_score = max(0, (0.5 - anomaly_score) * 2)
            
            return min(1.0, normalized_score)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return 0.5  # Default moderate anomaly score
    
    def _classify_threat(self, features: np.ndarray, event: ThreatEvent) -> Dict:
        """Classify the type of threat using ML and rule-based approaches"""
        try:
            # ML-based classification
            threat_probabilities = self.models['threat_classifier'].predict_proba(features)[0]
            threat_classes = ['benign', 'malware', 'lateral_movement', 'data_exfiltration', 'privilege_escalation']
            
            ml_classification = {
                'type': threat_classes[np.argmax(threat_probabilities)],
                'confidence': float(np.max(threat_probabilities)),
                'probabilities': {threat_classes[i]: float(prob) for i, prob in enumerate(threat_probabilities)}
            }
            
            # Rule-based pattern matching
            rule_based_threats = []
            
            # Check for APT indicators
            if self._contains_apt_indicators(event):
                rule_based_threats.append(('apt_campaign', 0.9))
            
            # Check for ransomware indicators
            if self._contains_ransomware_indicators(event):
                rule_based_threats.append(('ransomware', 0.95))
            
            # Check for insider threat patterns
            if self._contains_insider_threat_indicators(event):
                rule_based_threats.append(('insider_threat', 0.8))
            
            # Combine ML and rule-based results
            final_classification = ml_classification
            if rule_based_threats:
                best_rule_threat = max(rule_based_threats, key=lambda x: x[1])
                if best_rule_threat[1] > ml_classification['confidence']:
                    final_classification['type'] = best_rule_threat[0]
                    final_classification['confidence']
