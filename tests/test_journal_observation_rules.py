import unittest

from vitaldb_state_selection.anesthesia import (
    BISEvent,
    BISObservationProcessor,
    BISReason,
    ObservationRule,
    PreprocessingID,
    SQIEvent,
    SyntheticObservationTemplate,
)


class JournalObservationRuleTests(unittest.TestCase):
    def setUp(self):
        self.template = SyntheticObservationTemplate(
            "journal-rule-test",
            60.0,
            (BISEvent(0.0), BISEvent(10.0), BISEvent(20.0)),
            (SQIEvent(0.0, 80.0), SQIEvent(10.0, 40.0), SQIEvent(20.0, 80.0)),
        )

    def test_gate_and_age_are_independent(self):
        permissive20 = BISObservationProcessor(
            PreprocessingID.P0, self.template, ObservationRule("off_20", None, 20.0)
        )
        gated30 = BISObservationProcessor(
            PreprocessingID.P0, self.template, ObservationRule("sqi50_30", 50.0, 30.0)
        )
        for processor in (permissive20, gated30):
            processor.ingest(self.template.bis_events[0], 90.0)
            processor.ingest(self.template.bis_events[1], 70.0)
        self.assertEqual(permissive20.query(10.0).value, 70.0)
        self.assertEqual(gated30.query(10.0).value, 90.0)
        self.assertEqual(gated30.audit_events[-1].reason, BISReason.SQI_LOW)
        self.assertEqual(permissive20.query(21.0).mask, 1.0)

    def test_historical_defaults_are_unchanged(self):
        p0 = BISObservationProcessor(PreprocessingID.P0, self.template)
        p1 = BISObservationProcessor(PreprocessingID.P1, self.template)
        self.assertEqual((p0.staleness_cap, p1.staleness_cap), (30.0, 20.0))
        p0.ingest(self.template.bis_events[1], 70.0)
        p1.ingest(self.template.bis_events[1], 70.0)
        self.assertEqual(p0.audit_events[-1].reason, BISReason.AVAILABLE)
        self.assertEqual(p1.audit_events[-1].reason, BISReason.SQI_LOW)


if __name__ == "__main__":
    unittest.main()
