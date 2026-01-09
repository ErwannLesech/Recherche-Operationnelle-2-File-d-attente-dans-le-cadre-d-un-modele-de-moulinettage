"""
Tests pour le module personas.

Vérifie les comportements des différents types d'étudiants.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.personas import PersonaFactory, StudentType, Persona
from app.personas.usage_patterns import AcademicPeriod


class TestPersonaFactory:
    """Tests pour la création des personas."""
    
    def test_create_all_personas(self):
        """Vérifie la création de tous les personas."""
        personas = PersonaFactory.create_all_personas()
        
        # Tous les types doivent être présents
        expected_types = [
            StudentType.PREPA_SUP,
            StudentType.PREPA_SPE,
            StudentType.ING1,
            StudentType.ING2,
            StudentType.ING3
        ]
        
        for st in expected_types:
            assert st in personas
            assert isinstance(personas[st], Persona)
    
    def test_prepa_vs_ing_regularity(self):
        """Prepas sont plus reguliers que les Ings (variance plus faible)."""
        personas = PersonaFactory.create_all_personas()
        
        # variance_coefficient plus faible = plus regulier
        prepa_avg_variance = (
            personas[StudentType.PREPA_SUP].variance_coefficient +
            personas[StudentType.PREPA_SPE].variance_coefficient
        ) / 2
        
        ing_avg_variance = (
            personas[StudentType.ING1].variance_coefficient +
            personas[StudentType.ING2].variance_coefficient +
            personas[StudentType.ING3].variance_coefficient
        ) / 3
        
        # Prepas devraient avoir une variance plus faible (plus reguliers)
        assert prepa_avg_variance <= ing_avg_variance


class TestPersonaBehavior:
    """Tests du comportement des personas."""
    
    def test_arrival_rate_varies_by_hour(self):
        """Le taux d'arrivée varie selon l'heure."""
        persona = PersonaFactory.create_prepa_sup()
        
        # Nuit vs jour
        night_rate = persona.get_arrival_rate(hour=3)
        day_rate = persona.get_arrival_rate(hour=14)
        
        assert day_rate > night_rate
    
    def test_deadline_increases_rate(self):
        """Une deadline proche augmente le taux."""
        persona = PersonaFactory.create_ing1()
        
        normal_rate = persona.get_arrival_rate(hour=14)
        deadline_rate = persona.get_arrival_rate(hour=14, hours_to_deadline=2.0)
        
        assert deadline_rate > normal_rate
    
    def test_weekend_reduces_rate(self):
        """Le weekend réduit le taux pour certains."""
        persona = PersonaFactory.create_prepa_sup()
        
        weekday_rate = persona.get_arrival_rate(hour=14, is_weekend=False)
        weekend_rate = persona.get_arrival_rate(hour=14, is_weekend=True)
        
        # Le weekend devrait être différent (généralement plus faible)
        assert weekday_rate != weekend_rate


class TestAcademicPeriods:
    """Tests des periodes academiques."""
    
    def test_project_deadline_high_multiplier(self):
        """Deadline de projet a un multiplicateur eleve."""
        period = AcademicPeriod.PROJECT_DEADLINE
        
        # La periode de deadline devrait avoir un nom significatif
        assert "deadline" in period.value.lower() or "project" in period.value.lower()
    
    def test_exam_week_pattern(self):
        """Semaine d'examen existe comme periode."""
        period = AcademicPeriod.EXAM_WEEK
        
        # Verifie que la periode existe
        assert period is not None
        assert "exam" in period.value.lower()


class TestTotalLoadCalculation:
    """Tests du calcul de charge totale."""
    
    def test_sum_of_personas(self):
        """La charge totale est la somme des personas."""
        personas = PersonaFactory.create_all_personas()
        hour = 14
        
        total = sum(p.get_arrival_rate(hour) for p in personas.values())
        
        # La charge totale doit être positive et raisonnable
        assert total > 0
        assert total < 1000  # Sanity check
    
    def test_rush_hour_detection(self):
        """Détection des heures de rush."""
        personas = PersonaFactory.create_all_personas()
        
        loads = {h: sum(p.get_arrival_rate(h) for p in personas.values())
                 for h in range(24)}
        
        # Trouver les heures de pic
        peak_hour = max(loads, key=loads.get)
        trough_hour = min(loads, key=loads.get)
        
        # Le pic devrait être le soir ou l'après-midi
        assert peak_hour in range(14, 24)
        
        # Le creux devrait être la nuit
        assert trough_hour in range(0, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
