from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

from .buckets import observable_role
from .profiles import expected_profile
from .capture import CaptureDisposition, load_capabilities, capture_field_present
from .care_pathway import PatientPathway
from .ids import cohort_id, full_url, strong_identifier
from .metadata import meta
from .condition_coding import select_condition_coding
from .capture_class import capture_class_coding
from .clinical_semantics import order_compatible


ROLE_RESOURCE_TYPE = {
    "encounter": "Encounter", "emergency-encounter": "Encounter",
    "inpatient-encounter": "Encounter", "follow-up": "Encounter",
    "condition": "Condition", "assessment": "Observation",
    "order": "ServiceRequest", "referral": "ServiceRequest",
    "collection": "Specimen", "result": "Observation",
    "growth-measurement": "Observation", "prescription": "MedicationRequest",
    "dispense": "MedicationDispense", "medication-review": "MedicationStatement",
    "medication-administration": "MedicationAdministration",
    "procedure": "Procedure", "oncology-treatment": "Procedure",
    "immunization": "Immunization", "discharge": "DocumentReference",
}


def _timestamp(value: Any) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _coding(fact: Any) -> dict[str, str]:
    return {"system": fact.system, "code": fact.code, "display": fact.display}


def _resource(
    role: str, fact: Any, occurrence: Any, patient_url: str, practitioner_url: str,
    organization_url: str, prior_url: str | None, prior_type: str | None,
    disposition: CaptureDisposition,
) -> dict[str, Any]:
    coding = _coding(fact)
    start = _timestamp(occurrence.start)
    end = _timestamp(occurrence.end)
    if role in {"encounter", "emergency-encounter", "inpatient-encounter", "follow-up"}:
        class_code = "EMER" if role == "emergency-encounter" else "IMP" if role == "inpatient-encounter" else "AMB"
        resource = {
            "resourceType": "Encounter", "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": class_code},
            "type": [{"coding": [{"system": "urn:hdp-test-bundle:synthetic:encounter-type", "code": role}]}],
            "subject": {"reference": patient_url},
            "participant": [{"individual": {"reference": practitioner_url}}],
            "serviceProvider": {"reference": organization_url},
            "period": {"start": start, "end": _timestamp(occurrence.end + timedelta(minutes=30))},
            "reasonCode": [{"coding": [coding]}],
        }
        if prior_url and prior_type in {"Condition", "Procedure", "Observation"}:
            resource["reasonReference"] = [{"reference": prior_url}]
        if prior_url and prior_type == "ServiceRequest":
            resource["basedOn"] = [{"reference": prior_url}]
        if fact.family == "reproductive-care" and fact.status == "contraception-preconception-review":
            resource["reasonCode"][0]["text"] = "Contraception and preconception review"
        if fact.family == "diet-counselling":
            resource["reasonCode"][0]["text"] = "Diet and lifestyle counselling for type 2 diabetes"
        if fact.family == "pharmacist-review":
            resource["reasonCode"][0]["text"] = "Pharmacy medication review"
        if fact.family == "late-adherence-review":
            resource["reasonCode"][0]["text"] = "Reported refill gap and adherence review"
        return resource
    if role == "condition":
        status = "resolved" if fact.status == "resolved" else "active"
        resource = {
            "resourceType": "Condition",
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": status}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
            "code": {"coding": [coding]}, "subject": {"reference": patient_url},
            "onsetDateTime": start, "recordedDate": start,
        }
        if status == "resolved":
            resource["abatementDateTime"] = end
        return resource
    if role in {"assessment", "result", "growth-measurement"}:
        vital_codes = {"8302-2", "29463-7", "39156-5", "8480-6", "8462-4", "59408-5"}
        category = (
            {"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}
            if fact.code in vital_codes else
            {"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB"}
            if fact.system == "http://loinc.org" and fact.code not in {"24606-6", "19868-9"} else
            {"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}
        )
        resource = {
            "resourceType": "Observation", "status": "final",
            "category": [{"coding": [category]}],
            "code": {"coding": [coding]}, "subject": {"reference": patient_url},
            "effectiveDateTime": start, "issued": _timestamp(occurrence.end + timedelta(minutes=disposition.documentation_lag_minutes)),
            "performer": [{"reference": practitioner_url}],
        }
        if fact.code == "59408-5":
            resource["code"]["coding"].append({"system": "http://loinc.org", "code": "2708-6",
                                                   "display": "Oxygen saturation in Arterial blood"})
        if prior_url and prior_type == "ServiceRequest":
            resource["basedOn"] = [{"reference": prior_url}]
        elif prior_url and prior_type == "Specimen":
            resource["specimen"] = {"reference": prior_url}
        if fact.coded_value is not None:
            resource["valueCodeableConcept"] = {"coding": [fact.coded_value]}
            if fact.code != "82675-0":
                resource["referenceRange"] = [{"text": "Expected non-reactive qualitative result."}]
            resource["interpretation"] = [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "NEG" if fact.coded_value["code"] == "260385009" else "POS",
            }]}]
        if fact.code == "82675-0":
            resource["method"] = {"text": "NAA with probe detection"}
        if fact.family == "screening-eligibility":
            items = ["Asymptomatic at screening", "No prior abnormal screening result or relevant treatment"]
            if fact.status == "cervix-present":
                items += ["Cervix present", "No cervical screening in prior three years"]
            else:
                items += ["No routine mammogram in prior two years", "Average breast cancer risk"]
            resource["component"] = [{"code": {"text": item}, "valueBoolean": True} for item in items]
        if fact.family == "reproductive-assessment" and fact.status == "nonpregnant":
            resource["component"] = [{"code": {"text": "No current pregnancy reported"}, "valueBoolean": True}]
        if fact.value is not None:
            if fact.code in {"8480-6", "8462-4"}:
                if fact.blood_pressure_pair is None:
                    raise ValueError("blood pressure requires a jointly modeled pair")
                systolic, diastolic = fact.blood_pressure_pair
                resource["code"] = {"coding": [{
                    "system": "http://loinc.org", "code": "85354-9",
                    "display": "Blood pressure panel with all children optional",
                }]}
                resource["component"] = [
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
                        "valueQuantity": {"value": value, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                        "referenceRange": [{
                            "low": {"value": low, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                            "high": {"value": high, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                        }],
                        "interpretation": [{"coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": "N" if low <= value <= high else "H" if value > high else "L",
                        }]}],
                    }
                    for code, display, value, low, high in (
                        ("8480-6", "Systolic blood pressure", systolic, 90.0, 140.0),
                        ("8462-4", "Diastolic blood pressure", diastolic, 60.0, 90.0),
                    )
                ]
            else:
                resource["valueQuantity"] = {
                    "value": fact.value, "unit": fact.unit,
                    "system": "http://unitsofmeasure.org", "code": fact.unit,
                }
                if fact.reference_low is not None and fact.reference_high is not None:
                    resource["referenceRange"] = [{
                        "low": {"value": fact.reference_low, "unit": fact.unit, "system": "http://unitsofmeasure.org", "code": fact.unit},
                        "high": {"value": fact.reference_high, "unit": fact.unit, "system": "http://unitsofmeasure.org", "code": fact.unit},
                    }]
                if fact.interpretation is not None:
                    resource["interpretation"] = [{"coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": fact.interpretation,
                    }]}]
        elif fact.coded_value is None:
            if fact.result_text:
                resource["valueString"] = fact.result_text
            elif not resource.get("component"):
                resource["dataAbsentReason"] = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/data-absent-reason", "code": "unknown"}]}
                resource["note"] = [{"text": "Assessment/test occurrence captured; result content unavailable."}]
        return resource
    if role in {"order", "referral"}:
        resource = {
            "resourceType": "ServiceRequest", "status": fact.status if getattr(fact, "family", None) == "diabetic-eye-referral" else "completed", "intent": "order",
            "category": [{"coding": [{"system": "urn:hdp-test-bundle:synthetic:service-request-category", "code": role}]}],
            "code": {"coding": [coding]}, "subject": {"reference": patient_url},
            "authoredOn": start, "requester": {"reference": practitioner_url},
            "occurrenceDateTime": start,
        }
        if prior_url and prior_type in {"Condition", "Device", "Observation", "Medication", "Procedure", "DocumentReference"}:
            resource["supportingInfo"] = [{"reference": prior_url}]
        if getattr(fact, "family", None) == "diabetic-eye-referral":
            resource["code"] = {"text": "Diabetic eye assessment"}
            resource["performerType"] = {"text": "Ophthalmology"}
        return resource
    if role == "collection":
        resource = {
            "resourceType": "Specimen", "status": "available", "type": {"coding": [coding]},
            "subject": {"reference": patient_url}, "receivedTime": end,
            "collection": {"collector": {"reference": practitioner_url}, "collectedDateTime": start},
        }
        if prior_url and prior_type == "ServiceRequest":
            resource["request"] = [{"reference": prior_url}]
        if fact.family == "cervical-specimen":
            resource["type"] = {"text": "Cervical specimen"}
            resource["collection"]["bodySite"] = {"text": "Cervix"}
        return resource
    if role == "prescription":
        resource = {
            "resourceType": "MedicationRequest", "status": fact.status, "intent": "order",
            "medicationCodeableConcept": {"coding": [coding]}, "subject": {"reference": patient_url},
            "authoredOn": start, "requester": {"reference": practitioner_url},
            "dosageInstruction": [{"text": f"Synthetic reviewed therapy: {fact.display}"}],
        }
        if prior_url and prior_type == "Condition":
            resource["reasonReference"] = [{"reference": prior_url}]
        return resource
    if role == "dispense":
        resource = {
            "resourceType": "MedicationDispense", "status": "completed",
            "medicationCodeableConcept": {"coding": [coding]}, "subject": {"reference": patient_url},
            "performer": [{"actor": {"reference": practitioner_url}}], "whenHandedOver": start,
            "quantity": {"value": 30, "unit": "tablet", "system": "http://snomed.info/sct", "code": "428673006"},
            "daysSupply": {"value": 30, "unit": "day", "system": "http://unitsofmeasure.org", "code": "d"},
        }
        if prior_url:
            resource["authorizingPrescription"] = [{"reference": prior_url}]
        return resource
    if role == "medication-review":
        resource = {
            "resourceType": "MedicationStatement", "status": "stopped" if fact.status == "stopped" else "active",
            "medicationCodeableConcept": {"coding": [coding]}, "subject": {"reference": patient_url},
            "effectiveDateTime": start, "dateAsserted": _timestamp(occurrence.end + timedelta(minutes=disposition.documentation_lag_minutes)),
        }
        if prior_url:
            resource["derivedFrom"] = [{"reference": prior_url}]
        return resource
    if role == "medication-administration":
        return {
            "resourceType": "MedicationAdministration", "status": "completed",
            "medicationCodeableConcept": {"coding": [coding]}, "subject": {"reference": patient_url},
            "effectiveDateTime": start, "performer": [{"actor": {"reference": practitioner_url}}],
        }
    if role in {"procedure", "oncology-treatment"}:
        resource = {
            "resourceType": "Procedure", "status": "completed", "code": {"coding": [coding]},
            "subject": {"reference": patient_url}, "performedDateTime": start,
            "performer": [{"actor": {"reference": practitioner_url}}],
            **(
                {"basedOn": [{"reference": prior_url}]}
                if prior_url and prior_type in {"CarePlan", "ServiceRequest"}
                else {"reasonReference": [{"reference": prior_url}]}
                if prior_url and prior_type in {"Condition", "Observation", "DiagnosticReport", "DocumentReference"}
                else {}
            ),
        }
        if fact.result_text:
            resource["note"] = [{"text": fact.result_text}]
        if fact.family == "open-appendectomy":
            resource["code"]["text"] = "Open appendectomy"
        return resource
    if role == "immunization":
        return {
            "resourceType": "Immunization", "status": "completed", "vaccineCode": {"coding": [coding]},
            "patient": {"reference": patient_url}, "occurrenceDateTime": start,
            "primarySource": disposition.disposition == "performed",
            **({"lotNumber": f"SYNTHETIC-{fact.code}-{occurrence.start.year}"} if disposition.disposition == "performed" else {}),
            "performer": [{"actor": {"reference": practitioner_url}}],
        }
    if role == "discharge":
        resource = {
            "resourceType": "DocumentReference", "status": "current", "type": {"coding": [coding]},
            "subject": {"reference": patient_url}, "date": start,
            "author": [{"reference": practitioner_url}],
            "content": [{"attachment": {"contentType": "text/plain", "data": "U3ludGhldGljIGRpc2NoYXJnZSBzdW1tYXJ5Lg=="}}],
        }
        if prior_url:
            resource["context"] = {"related": [{"reference": prior_url}]}
        return resource
    raise ValueError(f"unsupported latent event role: {role}")


def _profile_fields(resource, event, fact, spec, source, practitioner_url, organization_url,
                    pathway, event_urls, event_types):
    """Populate the reviewed HDP shape, retaining identifier-only external context."""
    kind = resource["resourceType"]
    fact_by_id = {row.id: row for row in pathway.facts}
    related = [row for row in pathway.events if row.episode_id == event.episode_id]

    def reference(resource_type, roles, suffix):
        context_event = event
        if kind == "Observation" and resource_type == "Encounter":
            explicit_order = next((row for row in related if any(link.relationship == "result-of" and link.target_event_id == row.id for link in event.links)), None)
            context_event = explicit_order or event
        if kind == "Observation" and resource_type in {"ServiceRequest", "Specimen"}:
            linked_ids = {link.target_event_id for link in event.links}
            target = next((row for row in related if row.id in linked_ids and row.role in roles
                           and (resource_type != "ServiceRequest" or order_compatible(fact.code, fact_by_id[row.fact_id].code))), None)
        else:
            target = next((row for row in reversed(related) if row.role in roles
                           and row.occurrence.start <= context_event.occurrence.start and row.id != event.id), None)
        key = target.id if target else f"{event.episode_id if suffix == 'encounter' else event.id}-{suffix}"
        result = {"type": resource_type, "identifier": {
            "system": f"urn:hdp-test-bundle:synthetic:{source}:{resource_type.lower()}", "value": key,
        }}
        if target and target.id in event_urls and event_types[target.id] == resource_type:
            result["reference"] = event_urls[target.id]
            result["identifier"] = strong_identifier(spec, source, resource_type, event_urls[target.id].removeprefix("urn:uuid:"))
        return result

    subject = resource.get("subject", resource.get("patient"))
    if subject:
        subject["identifier"] = {"system": spec["identity"]["cohortIdentifierSystem"], "value": pathway.cohort_id}
    indication = fact_by_id.get(event.indication_fact_id, fact)
    if kind == "Condition":
        resource["recorder"] = {"reference": practitioner_url}
    elif kind == "Encounter":
        profile = expected_profile(resource, source)
        encounter_type = (
            "hospitalization-hdp" if profile.endswith("/il-hdp-encounter-hospitalization")
            else "community-hmo-ambulatory-hdp" if profile.endswith("/il-hdp-encounter-community-hmo-ambulatory")
            else "supplemental-encounter-hdp"
        )
        resource["type"] = [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-encounter-type", "code": encounter_type}]}]
        resource["serviceType"] = {"coding": [{"system": "http://snomed.info/sct", "code": "700232004", "display": "General medical service"}]}
        resource["participant"][0]["type"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType", "code": "PPRF", "display": "primary performer"}]}]
        reason = reference("Condition", {"condition"}, "encounter-indication")
        reason["display"] = indication.display
        resource.setdefault("reasonReference", [reason])
        resource["diagnosis"] = [{"condition": reason, "use": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/diagnosis-role", "code": "AD", "display": "Admission diagnosis"}]}}]
    elif kind == "Observation" and expected_profile(resource, source).endswith(
        ("/il-hdp-observation-lab", "/il-core-observation-lab")
    ):
        resource["encounter"] = reference("Encounter", {"encounter", "emergency-encounter", "inpatient-encounter", "follow-up"}, "encounter")
        resource["category"].append({"coding": [{"system": "http://snomed.info/sct", "code": "108252007", "display": "Laboratory procedure"}]})
        if expected_profile(resource, source).endswith("/il-hdp-observation-lab"):
            order_reference = reference("ServiceRequest", {"order"}, "laboratory-order")
            resource["basedOn"] = [order_reference]
            for item in resource["basedOn"]:
                item.setdefault("identifier", order_reference["identifier"])
            resource["specimen"] = reference("Specimen", {"collection"}, "specimen")
    elif kind == "Observation" and fact.code == "59408-5":
        resource["encounter"] = reference("Encounter", {"encounter", "emergency-encounter", "inpatient-encounter", "follow-up"}, "encounter")
    elif kind == "ServiceRequest":
        if fact.family in {"diabetic-eye-referral", "diet-follow-up-order", "refill-follow-up-order"}:
            indication = next((fact_by_id[row.fact_id] for row in related if row.role == "condition"
                               and fact_by_id[row.fact_id].code == "44054006"), indication)
        category_code, category_display = (
            ("306206005", "Referral to specialist") if event.role == "referral" or fact.family in {"diet-follow-up-order", "refill-follow-up-order"} else
            ("387713003", "Surgical procedure") if fact.code in {"54885007", "387713003"} else
            ("108252007", "Laboratory procedure")
        )
        resource["category"] = [{"coding": [{"system": "http://snomed.info/sct", "code": category_code, "display": category_display}]}]
        resource["priority"] = "routine"
        resource["encounter"] = reference("Encounter", {"encounter", "emergency-encounter", "inpatient-encounter", "follow-up"}, "encounter")
        resource["performer"] = [{"reference": organization_url}]
        resource["reasonCode"] = [{"coding": [_coding(indication)]}]
        resource["reasonReference"] = [reference("Condition", {"condition"}, "indication")]
        resource["note"] = [{"text": f"Synthetic {event.role}: {indication.display}."}]
    elif kind == "Specimen":
        resource["request"] = [reference("ServiceRequest", {"order"}, "laboratory-order")]
        if fact.family != "cervical-specimen":
            resource["collection"]["bodySite"] = {"coding": [{"system": "http://snomed.info/sct", "code": "368208006", "display": "Left arm"}]}
    elif kind.startswith("Medication"):
        # Structured detail is required by the mapped HDP profiles. Lifecycle and
        # ingredient reconciliation is audited separately against clinical truth.
        detail = fact.medication_detail
        if detail is None:
            raise ValueError(f"missing structured regimen for {fact.code}")
        dose = detail["doseMg"]
        period = {"start": detail["exposureStart"]}
        if detail.get("exposureEnd"):
            period["end"] = detail["exposureEnd"]
        dosage = {"text": f"Synthetic documented dose: {dose:g} mg",
                  "timing": {"repeat": {"frequency": detail["frequencyPerDay"], "period": 1, "periodUnit": "d", "boundsPeriod": period}},
                  "route": {"coding": [{"system": "http://snomed.info/sct", "code": detail["routeCode"], "display": detail.get("routeDisplay", "Oral route")}]},
                  "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg"}}]}
        if fact.code in {"6809", "1100699"} and pathway.variant == "dispense-follow-up":
            # The ingredient coding stays RxNorm 6809. The product strength and
            # tablet count are separately stated, so the supply can be checked
            # from the emitted dosage without trusting a 30-day label.
            resource["medicationCodeableConcept"]["text"] = f"{fact.display} {dose:g} mg per tablet"
            dosage["text"] = f"Take one {dose:g} mg {fact.display} tablet by mouth once daily"
            dosage["doseAndRate"].append({"doseQuantity": {
                "value": 1, "unit": "tablet", "system": "http://unitsofmeasure.org", "code": "1",
            }})
        if kind == "MedicationStatement":
            resource.pop("effectiveDateTime", None)
            resource["effectivePeriod"] = period
            if detail.get("reason"):
                resource["statusReason"] = [{"text": detail["reason"]}]
        if detail.get("workflowAction"):
            resource["note"] = [{"text": "Documented medication workflow action: " + detail["workflowAction"]}]
        course = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-course-of-therapy", "code": "continuous", "display": "Continuous long term therapy"}]}
        if kind in {"MedicationRequest", "MedicationDispense"}:
            resource["dosageInstruction"] = [dosage]
        if kind == "MedicationRequest":
            resource["courseOfTherapyType"] = course
            if fact.code in {"6809", "1100699"}:
                resource["dispenseRequest"] = {"quantity": {"value": 30, "unit": "tablet", "system": "http://snomed.info/sct", "code": "428673006"},
                                               "expectedSupplyDuration": {"value": 30, "unit": "day", "system": "http://unitsofmeasure.org", "code": "d"}}
                if pathway.variant == "dispense-follow-up":
                    renewed = any(fact_by_id[row.fact_id].family == "metformin-renewal" for row in related)
                    resource["dispenseRequest"]["numberOfRepeatsAllowed"] = detail.get("plannedRepeats", 0)
                    resource["dispenseRequest"]["validityPeriod"] = {
                        "start": detail["exposureStart"],
                        **({"end": detail["authorizationEnd"]} if detail.get("authorizationEnd") else {}),
                    }
                    if fact.family == "metformin-renewal":
                        resource["dispenseRequest"]["numberOfRepeatsAllowed"] = detail.get("plannedRepeats", 0)
                        resource["statusReason"] = {"text": detail["reason"]}
                    elif renewed:
                        resource["statusReason"] = {"text": "Authorization exhausted after one 30-tablet fill; renewed for continued stable therapy"}
            resource["category"] = [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo" if source == "hmo" else "inpatient", "display": "Community Care/HMO" if source == "hmo" else "Inpatient"}]}]
        elif kind == "MedicationStatement":
            resource["dosage"] = [dosage]
            resource["extension"] = [{"url": "http://fhir.health.gov.il/StructureDefinition/ext-medication-course-of-therapy-type", "valueCodeableConcept": course}]
        elif kind == "MedicationAdministration":
            resource["dosage"] = {"text": dosage["text"], "route": dosage["route"], "dose": dosage["doseAndRate"][0]["doseQuantity"]}
        elif kind == "MedicationDispense":
            resource["type"] = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": detail.get("fillType", "FFC"), "display": "Refill - Complete" if detail.get("fillType") == "RFC" else "First Fill - Complete"}]}
    elif kind == "DocumentReference":
        resource["authenticator"] = {"reference": practitioner_url}
        resource["context"] = {"encounter": [reference("Encounter", {"encounter", "inpatient-encounter"}, "encounter")],
                               "period": {"start": _timestamp(event.occurrence.start), "end": _timestamp(event.occurrence.end)}}
        resource["content"][0]["attachment"]["creation"] = _timestamp(event.occurrence.start)


def _capture_fields(resource, disposition, spec, source, event, fact):
    owner = disposition.origin_source or source
    def actor(kind):
        url = full_url(spec["datasetVersion"], owner, "master", kind)
        result = {"type": kind, "identifier": strong_identifier(spec, owner, kind, url.removeprefix("urn:uuid:"))}
        if owner == source:
            result["reference"] = url
        return result
    practitioner, organization = actor("Practitioner"), actor("Organization")
    kind = resource["resourceType"]
    if "serviceProvider" in resource:
        resource["serviceProvider"] = organization
    if "participant" in resource:
        for participant in resource["participant"]:
            participant["individual"] = practitioner
    if "recorder" in resource:
        resource["recorder"] = practitioner
    if "requester" in resource:
        resource["requester"] = practitioner
    if "performer" in resource:
        resource["performer"] = [({"actor": practitioner} if "actor" in row else organization)
                                 for row in resource["performer"]]
    if "author" in resource:
        resource["author"] = [practitioner]
    if "authenticator" in resource:
        resource["authenticator"] = practitioner
    if kind == "MedicationStatement":
        resource["informationSource"] = organization
    if kind == "Specimen":
        resource["collection"]["collector"] = practitioner
    if kind == "Encounter":
        profile = expected_profile(resource, owner)
        code = "hospitalization-hdp" if profile.endswith("/il-hdp-encounter-hospitalization") else "community-hmo-ambulatory-hdp" if profile.endswith("/il-hdp-encounter-community-hmo-ambulatory") else "supplemental-encounter-hdp"
        resource["type"] = [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-encounter-type", "code": code}]}]
    if kind == "MedicationRequest":
        resource["category"] = [{"coding": [{"system": "http://fhir.health.gov.il/cs/il-core-medication-request-category", "code": "community-hmo" if owner == "hmo" else "inpatient", "display": "Community Care/HMO" if owner == "hmo" else "Inpatient"}]}]
    if disposition.disposition == "summarized":
        # Retain clinically useful facts and mandatory business data; omit the
        # detailed local workflow fields that a summary does not reproduce.
        for field in ("issued", "note", "supportingInfo", "method", "bodySite"):
            resource.pop(field, None)
        if kind == "Encounter" and expected_profile(resource, owner).endswith("/il-hdp-encounter-supplemental"):
            resource.pop("participant", None)
        if kind == "MedicationDispense":
            resource.pop("quantity", None)
    elif disposition.disposition == "referral-only":
        resource["status"] = "active"
        resource["performer"] = [{"type": "Organization", "identifier": strong_identifier(
            spec, source, "Organization", full_url(spec["datasetVersion"], source, "master", "Organization").removeprefix("urn:uuid:"))}]
        resource["note"] = [{"text": "Referral received; no local performance of the requested service is asserted."}]
    return resource


def _metadata_only(event, patient_url, practitioner_url):
    # The document index proves a record exists without carrying its clinical
    # code/result. The owner source retains the full event in its own view.
    return {
        "resourceType": "DocumentReference", "status": "current",
        "type": {"text": "External clinical record metadata"},
        "subject": {"reference": patient_url}, "date": _timestamp(event.occurrence.start),
        "author": [{"reference": practitioner_url}], "authenticator": {"reference": practitioner_url},
        "context": {"period": {"start": _timestamp(event.occurrence.start), "end": _timestamp(event.occurrence.end)},
                    "encounter": [{"type": "Encounter", "identifier": {"system": "urn:hdp-test-bundle:clinical-episode", "value": event.episode_id}}]},
        "description": "Clinical content withheld; metadata only.",
        "content": [{"attachment": {"contentType": "application/fhir+json", "url": f"urn:hdp-test-bundle:withheld-record:{event.id}",
                                      "creation": _timestamp(event.occurrence.start)}}],
    }


def project_pathway(
    spec: dict[str, Any], organization: dict[str, Any], index: int, patient_url: str,
    pathway: PatientPathway, captures: list[CaptureDisposition],
    condition_coding_by_event: dict[str, dict[str, str]] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Project source-specific FHIR from truth and return an exact mapping ledger."""
    source = organization["id"]
    capability = load_capabilities(spec["remediation"]["capturePolicyCatalog"])[source]
    facts = {row.id: row for row in pathway.facts}
    dispositions = {row.latent_event_id: row for row in captures}
    practitioner_url = full_url(spec["datasetVersion"], source, "master", "Practitioner")
    organization_url = full_url(spec["datasetVersion"], source, "master", "Organization")
    resources: list[tuple[str, dict[str, Any]]] = []
    mappings: list[dict[str, Any]] = []
    event_urls: dict[str, str] = {}
    event_types: dict[str, str] = {}
    for event in pathway.events:
        disposition = dispositions[event.id]
        if disposition.disposition == "absent":
            continue
        resource_type = ROLE_RESOURCE_TYPE[event.role]
        url = full_url(spec["datasetVersion"], source, cohort_id(index), resource_type, event.id)
        prior_event_id = event.links[0].target_event_id if event.links else None
        prior_url = event_urls.get(prior_event_id) if prior_event_id else None
        prior_type = event_types.get(prior_event_id) if prior_event_id else None
        resource = _resource(
            event.role, facts[event.fact_id], event.occurrence, patient_url,
            practitioner_url, organization_url, prior_url, prior_type, disposition,
        )
        if resource_type == "Condition":
            from .clinical_state import acute_snapshot
            acute_snapshot(resource, spec["timelineAnchor"])
        _profile_fields(resource, event, facts[event.fact_id], spec, source, practitioner_url, organization_url,
                        pathway, event_urls, event_types)
        fact = facts[event.fact_id]
        for link in event.links:
            if link.relationship == "supersedes" and resource_type == "MedicationRequest":
                target = next(row for row in pathway.events if row.id == link.target_event_id)
                if target.role == "prescription":
                    resource["priorPrescription"] = ({"reference": event_urls[target.id]}
                        if target.id in event_urls else {"type": "MedicationRequest", "identifier": {
                            "system": "urn:hdp-test-bundle:clinical-event-key", "value": target.id}})
            if link.relationship == "part-of":
                target = next(row for row in pathway.events if row.id == link.target_event_id)
                target_type = ROLE_RESOURCE_TYPE[target.role]
                reference = {"type": target_type, "identifier": {"system": "urn:hdp-test-bundle:clinical-event-key", "value": target.id}}
                if dispositions[target.id].disposition != "absent":
                    reference["reference"] = full_url(spec["datasetVersion"], source, cohort_id(index), target_type, target.id)
                resource["partOf"] = [reference]
        if fact.code == "5018-7" and event.role == "result":
            resource["specimen"] = {"type": "Specimen", "identifier": {
                "system": "urn:hdp-test-bundle:clinical-sample", "value": event.id}}
            resource["method"] = {"text": "HIV-1 RNA nucleic acid amplification"}
            resource.pop("derivedFrom", None)
        if fact.status == "unsupported-age-method" and resource_type == "Observation":
            for field in ("valueQuantity", "valueString", "referenceRange", "interpretation"):
                resource.pop(field, None)
            resource["dataAbsentReason"] = {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/data-absent-reason", "code": "not-performed"}]}
            resource["status"] = "cancelled"
            resource["note"] = [{"text": "Spirometry not performed: no supported method for this event age."}]
        if fact.code == "82675-0" and event.role == "result":
            if prior_event_id:
                resource["specimen"] = {"reference": prior_url} if prior_url else {
                    "type": "Specimen", "identifier": {"system": "urn:hdp-test-bundle:clinical-event-key", "value": prior_event_id},
                }
            order_event = next((row for row in pathway.events if row.episode_id == event.episode_id
                                and row.role == "order" and facts[row.fact_id].code == "82675-0"), None)
            if order_event:
                order_url = event_urls.get(order_event.id)
                resource["basedOn"] = [{"reference": order_url}] if order_url else [{
                    "type": "ServiceRequest", "identifier": {"system": "urn:hdp-test-bundle:clinical-event-key", "value": order_event.id},
                }]
        if fact.family == "screening-result" and prior_event_id and not resource.get("basedOn"):
            resource["basedOn"] = [{"reference": prior_url}] if prior_url else [{
                "type": "ServiceRequest", "identifier": {"system": "urn:hdp-test-bundle:clinical-event-key", "value": prior_event_id},
            }]
        if fact.family == "screen-result-review" and prior_event_id:
            resource["reasonReference"] = [{"reference": prior_url}] if prior_url else [{
                "type": "Observation", "identifier": {"system": "urn:hdp-test-bundle:clinical-event-key", "value": prior_event_id},
            }]
        if fact.family == "diabetic-eye-review" and prior_event_id:
            resource["basedOn"] = [{"reference": prior_url}] if prior_url else [{
                "type": "ServiceRequest", "identifier": {"system": "urn:hdp-test-bundle:clinical-event-key", "value": prior_event_id},
            }]
        if fact.family == "diet-follow-up-order":
            latest_hba1c = next((row for row in reversed(pathway.events)
                                 if row.episode_id == event.episode_id
                                 and row.role == "result" and facts[row.fact_id].code == "4548-4"
                                 and row.id in event_urls), None)
            if latest_hba1c:
                resource["supportingInfo"] = [{"reference": event_urls[latest_hba1c.id]}]
        if fact.family == "diet-follow-up":
            diabetes = next((row for row in pathway.events if row.episode_id == event.episode_id
                             and row.role == "condition" and facts[row.fact_id].code == "44054006"
                             and row.id in event_urls), None)
            resource["reasonReference"] = ([{"reference": event_urls[diabetes.id]}] if diabetes else [])
            resource["diagnosis"] = ([{"condition": {"reference": event_urls[diabetes.id]}}] if diabetes else [])
        if fact.family == "short-stay-community-review":
            surgery = next((row for row in pathway.events if row.episode_id == event.episode_id
                            and row.role == "procedure" and row.id in event_urls), None)
            resource["reasonReference"] = ([{"reference": event_urls[surgery.id]}] if surgery else [])
        if fact.family == "metformin-renewal":
            original = next((row for row in pathway.events if row.episode_id == event.episode_id
                             and row.role == "prescription" and facts[row.fact_id].family == "metformin"
                             and row.id in event_urls), None)
            diabetes = next((row for row in pathway.events if row.episode_id == event.episode_id
                             and row.role == "condition" and facts[row.fact_id].code == "44054006"
                             and row.id in event_urls), None)
            if original:
                resource["priorPrescription"] = {"reference": event_urls[original.id]}
            if diabetes:
                resource["reasonReference"] = [{"reference": event_urls[diabetes.id]}]
        if fact.family in {"pharmacist-reconciliation", "late-resumption"}:
            prescription = next((row for row in reversed(pathway.events)
                                 if row.episode_id == event.episode_id and row.role == "prescription"
                                 and row.occurrence.start <= event.occurrence.start and row.id in event_urls), None)
            if prescription:
                resource["derivedFrom"] = [{"reference": event_urls[prescription.id]}]
            if prior_url and prior_type == "Encounter":
                resource["context"] = {"reference": prior_url}
        if disposition.disposition == "metadata-only":
            resource = _metadata_only(event, patient_url, practitioner_url)
            resource_type = "DocumentReference"
            url = full_url(spec["datasetVersion"], source, cohort_id(index), resource_type, event.id)
        if event.sensitivity_fact and prior_url:
            if resource_type == "Observation":
                resource['derivedFrom'] = [{'reference': prior_url}]
            elif resource_type == "Condition":
                resource['evidence'] = [{'detail': [{'reference': prior_url}]}]
        _capture_fields(resource, disposition, spec, source, event, facts[event.fact_id])
        profile = expected_profile(resource, source)
        if profile is None:
            raise ValueError(f"no reviewed profile for pathway {event.role}/{resource_type}")
        semantic_role = observable_role(resource, source)
        resource["meta"] = meta(spec, source, profile, semantic_role, archetype=pathway.archetype)
        if event.sensitivity_fact:
            from .sensitivity import fact_rule
            rule = fact_rule(spec, event.sensitivity_fact)
            resource["meta"] = meta(spec, source, profile, semantic_role, archetype=pathway.archetype,
                confidentiality=rule['confidentiality'], sensitivity=rule['clinicalSensitivity'], nopat=rule['nopat'])
            fact_role = "hiv-result" if "hiv" in event.sensitivity_fact else {
                "mental-health-diagnosis": "mental-health-condition",
                "protected-domestic-violence-indicator": "violence-flag", "ordinary-injury": "ordinary-injury",
            }[event.sensitivity_fact]
            resource['meta']['tag'].extend([
                {"system": "urn:hdp-test-bundle:synthetic-sensitivity-fact", "code": event.sensitivity_fact},
                {"system": "urn:hdp-test-bundle:synthetic-fact-role", "code": fact_role}])
        resource["meta"]["tag"].append(capture_class_coding(spec, source, disposition.disposition))
        resource["meta"]["source"] = f"urn:hdp-test-bundle:source:{source}"
        if condition_coding_by_event and event.id in condition_coding_by_event:
            if resource_type != "Condition":
                raise ValueError("Condition coding selection targets a non-Condition event")
            resource = select_condition_coding(resource, condition_coding_by_event[event.id], spec["metadata"])
        resource["identifier"] = [strong_identifier(spec, source, resource_type, url.removeprefix("urn:uuid:"))]
        resource["identifier"].append({
            "system": "urn:hdp-test-bundle:clinical-event-key", "value": event.id,
        })
        resource["text"] = {
            "status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml">Synthetic {disposition.disposition} {event.role}.</div>',
        }
        for field in ("note", "text", "supportingInfo"):
            if field not in capability.optional_fields and field not in capability.mandatory_fields:
                resource.pop(field, None)
        missing = [field for field in capability.mandatory_fields if not capture_field_present(resource, field)]
        if missing:
            raise ValueError(f"{source}/{url}: missing mandatory capture fields {missing}")
        resources.append((url, resource))
        event_urls[event.id] = url
        event_types[event.id] = resource_type
        mappings.append({
            "rowType": "fhir-mapping", "cohortId": pathway.cohort_id,
            "pathwayId": pathway.id, "episodeId": event.episode_id,
            "eventId": event.id, "factId": event.fact_id,
            "source": source, "disposition": disposition.disposition,
            "provenanceRequired": disposition.provenance_required,
            "originSource": disposition.origin_source or source,
            "resourceFullUrl": url, "resourceType": resource_type,
            "observableCode": asdict(facts[event.fact_id]),
        })
    return resources, mappings
