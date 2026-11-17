PROFILES = {
    "Patient": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-patient",
    "Condition": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-condition",
    "Observation": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-observation-lab",
    "Organization": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-organization",
    "Practitioner": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-practitioner",
    "PractitionerRole": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-practitioner-role",
}

ENCOUNTER_PROFILES = {
    "hmo": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-encounter-community-hmo-ambulatory",
    "hospital-a": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-encounter-hospitalization",
    "hospital-b": "http://hdp.fhir.health.gov.il/StructureDefinition/il-hdp-encounter-hospitalization",
}

DEFAULT_BUCKETS = {
    "AllergyIntolerance": "allergiesAndIntolerances",
    "CarePlan": "recommendations",
    "CareTeam": "encounterInformation",
    "Communication": "clinicalNotes",
    "CommunicationRequest": "clinicalNotes",
    "Condition": "diagnoses",
    "Device": "procedures",
    "DeviceRequest": "recommendations",
    "DiagnosticReport": "laboratoryTests",
    "DocumentReference": "clinicalNotes",
    "Encounter": "encounterInformation",
    "EpisodeOfCare": "encounterInformation",
    "Flag": "clinicalNotes",
    "Goal": "assessments",
    "ImagingStudy": "diagnosticImaging",
    "Immunization": "immunizations",
    "ImmunizationRecommendation": "immunizations",
    "Location": "organization",
    "Measure": "assessments",
    "MeasureReport": "assessments",
    "Media": "diagnosticImaging",
    "Medication": "medications",
    "MedicationAdministration": "medications",
    "MedicationDispense": "medications",
    "MedicationRequest": "medications",
    "MedicationStatement": "medications",
    "MolecularSequence": "genetics",
    "NutritionOrder": "recommendations",
    "Observation": "measurements",
    "Organization": "organization",
    "OrganizationAffiliation": "organization",
    "Patient": "patientDemographics",
    "Practitioner": "practitioner",
    "PractitionerRole": "practitioner",
    "Procedure": "procedures",
    "Questionnaire": "assessments",
    "QuestionnaireResponse": "assessments",
    "RelatedPerson": "patientDemographics",
    "ResearchStudy": "participationInResearch",
    "ResearchSubject": "participationInResearch",
    "ServiceRequest": "referrals",
    "Specimen": "laboratoryTests",
    "Substance": "pathology",
    "SupplyDelivery": "procedures",
}


HDP = "http://hdp.fhir.health.gov.il/StructureDefinition/"
IL_CORE = "http://fhir.health.gov.il/StructureDefinition/"

# The role is inferred from FHIR content, never from meta.profile or generator tags.
RESOURCE_PROFILES = {
    **PROFILES,
    "AllergyIntolerance": HDP + "il-hdp-allergy-intolerance",
    "RelatedPerson": HDP + "il-hdp-related-person",
    "MedicationRequest": HDP + "il-hdp-medication-request",
    "MedicationStatement": HDP + "il-hdp-medication-statement",
    "MedicationAdministration": HDP + "il-hdp-medication-administration",
    "MedicationDispense": HDP + "il-hdp-medication-dispense",
    "DocumentReference": HDP + "il-hdp-document-reference",
    "ServiceRequest": HDP + "il-hdp-service-request",
    "Specimen": HDP + "il-hdp-specimen-lab",
    "Procedure": IL_CORE + "il-core-procedure",
    "Immunization": IL_CORE + "il-core-immunization",
}
LAB_CODES = {"6298-4", "2160-0", "718-7", "4548-4", "2345-7", "2093-3", "2085-9", "13457-7", "1742-6", "630-4"}
VITAL_CODES = {"8302-2", "29463-7", "39156-5", "8480-6", "8462-4", "85354-9", "8867-4", "8310-5", "9279-1", "2708-6"}


def expected_profile(resource: dict, source: str = "hmo") -> str | None:
    kind = resource["resourceType"]
    if kind == "Observation":
        codes = {c.get("code") for c in resource.get("code", {}).get("coding", [])}
        categories = {c.get("code") for row in resource.get("category", []) for c in row.get("coding", [])}
        # The HDP lab profile requires a numeric low and high reference range;
        # Qualitative HIV antibody is still a laboratory result, but the HDP
        # lab profile requires numeric reference-range bounds. IL Core lab
        # permits a text range and retains the laboratory bucket label.
        if codes & {"68961-2", "5018-7", "58410-2"}:
            return IL_CORE + "il-core-observation-lab"
        # Qualitative cervical HPV DNA follows the general HDP Observation
        # profile with the assessment bucket.
        if "82675-0" in codes:
            return HDP + "il-hdp-observation"
        if "85354-9" in codes:
            return HDP + "il-hdp-bp"
        if codes & VITAL_CODES or "vital-signs" in categories:
            return HDP + "il-hdp-vital-signs"
        if codes & LAB_CODES or categories & {"LAB", "laboratory", "108252007"}:
            return HDP + "il-hdp-observation-lab"
        if "31044-1" in codes:
            return IL_CORE + "il-core-immunization-reaction-observation"
        if "75325-1" in codes:
            return HDP + "il-hdp-patient-reported-chief-complaint"
        return HDP + "il-hdp-observation"
    if kind == "Encounter":
        from .capture import observable_origin
        source = observable_origin(resource, source)
        type_codes = {c.get("code") for row in resource.get("type", []) for c in row.get("coding", [])}
        if "inpatient-care-segment-hdp" in type_codes or "partOf" in resource:
            return HDP + "il-hdp-encounter-inpatient-care-segment"
        if resource.get("class", {}).get("code") == "IMP":
            return HDP + "il-hdp-encounter-hospitalization"
        if source == "hmo":
            return ENCOUNTER_PROFILES["hmo"]
        return HDP + "il-hdp-encounter-supplemental"
    if kind == "Specimen" and resource.get("collection", {}).get("method", {}).get("coding", [{}])[0].get("code") == "129314006":
        return HDP + "il-hdp-specimen-pathology"
    return RESOURCE_PROFILES.get(kind)
