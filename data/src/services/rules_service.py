"""In-memory rules service used by API contract tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class RuleDefinition:
    rule_id: str
    name: str
    type: str
    description: Optional[str] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    severity: Optional[str] = None
    action: Optional[str] = None
    enabled: bool = True
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RulesService:
    """Simple in-memory rule registry backing API contract tests."""

    def __init__(self) -> None:
        self._rules: Dict[str, RuleDefinition] = {}
        self._rule_counter = 1

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def create_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        rule_id = f"rule_{self._rule_counter:05d}"
        self._rule_counter += 1

        definition = RuleDefinition(
            rule_id=rule_id,
            name=rule_data.get("name", rule_id),
            type=rule_data.get("type", "validation"),
            description=rule_data.get("description"),
            conditions=rule_data.get("conditions", []),
            severity=rule_data.get("severity"),
            action=rule_data.get("action"),
            enabled=bool(rule_data.get("enabled", True)),
            metadata={"conditions_count": len(rule_data.get("conditions", []))},
        )
        self._rules[rule_id] = definition

        return {
            "rule_id": rule_id,
            "name": definition.name,
            "type": definition.type,
            "status": "created",
            "version": definition.version,
            "created_at": definition.created_at.isoformat() + "Z",
            "created_by": definition.created_by,
            "conditions_count": len(definition.conditions or []),
            "validation_passed": True,
        }

    def get_rules(self, page: int = 1, page_size: int = 20, *, enabled: Optional[bool] = None) -> Dict[str, Any]:
        rules = list(self._rules.values())
        if enabled is not None:
            rules = [rule for rule in rules if rule.enabled == enabled]

        start = (page - 1) * page_size
        end = start + page_size
        paginated = rules[start:end]

        return {
            "rules": [self._serialize_rule(rule) for rule in paginated],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_rules": len(rules),
                "total_pages": max(1, (len(rules) + page_size - 1) // page_size),
                "has_next": end < len(rules),
                "has_prev": start > 0,
            },
            "filters": {
                "type": "all",
                "enabled": enabled if enabled is not None else True,
                "search_query": None,
            },
        }

    def get_rule(self, rule_id: str) -> Dict[str, Any]:
        rule = self._rules.get(rule_id)
        if not rule:
            raise KeyError(f"Rule '{rule_id}' not found")
        return self._serialize_rule(rule)

    def update_rule(self, rule_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        rule = self._rules.get(rule_id)
        if not rule:
            raise KeyError(f"Rule '{rule_id}' not found")

        changes = []
        if "name" in update_data and update_data["name"] != rule.name:
            changes.append({"field": "name", "old_value": rule.name, "new_value": update_data["name"]})
            rule.name = update_data["name"]

        if "description" in update_data:
            changes.append({"field": "description", "old_value": rule.description, "new_value": update_data["description"]})
            rule.description = update_data["description"]

        if "conditions" in update_data:
            old_count = len(rule.conditions or [])
            new_conditions = update_data["conditions"] or []
            changes.append({"field": "conditions", "old_count": old_count, "new_count": len(new_conditions)})
            rule.conditions = new_conditions

        rule.enabled = bool(update_data.get("enabled", rule.enabled))
        rule.version += 1

        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "type": rule.type,
            "version": rule.version,
            "status": "updated",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "changes": changes,
            "validation_result": {
                "syntax_valid": True,
                "semantic_valid": True,
                "performance_impact": "low",
            },
        }

    def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        rule = self._rules.pop(rule_id, None)
        if not rule:
            raise KeyError(f"Rule '{rule_id}' not found")

        return {
            "rule_id": rule_id,
            "status": "deleted",
            "deleted_at": datetime.utcnow().isoformat() + "Z",
            "deleted_by": "system",
            "backup_created": True,
            "backup_location": f"/backups/rules/{rule_id}.json",
            "affected_datasets": [],
            "cascade_effects": "none",
        }

    # ------------------------------------------------------------------
    # Rule evaluation helpers
    # ------------------------------------------------------------------
    def test_rule(self, rule_id: str, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        rule = self._rules.get(rule_id)
        if not rule:
            raise KeyError(f"Rule '{rule_id}' not found")

        total = len(sample_data)
        failures = []
        for index, record in enumerate(sample_data):
            price = record.get("price")
            if price is None or price <= 0 or price >= 10000:
                failures.append({
                    "record_index": index,
                    "field": "price",
                    "value": price,
                    "violation": "Price outside expected range",
                    "severity": "error",
                })

        passed = total - len(failures)
        return {
            "rule_id": rule_id,
            "test_status": "completed",
            "test_results": {
                "total_records": total,
                "passed_records": passed,
                "failed_records": len(failures),
                "pass_rate": passed / total if total else 0,
                "failures": failures,
            },
            "performance_metrics": {
                "execution_time_ms": 5.0,
                "memory_usage_bytes": 512,
                "records_per_second": total / 0.01 if total else 0,
            },
            "recommendations": [],
        }

    def bulk_operation(self, operation: str, rule_ids: List[str], dry_run: bool = False) -> Dict[str, Any]:
        processed = []
        for rule_id in rule_ids:
            rule = self._rules.get(rule_id)
            if not rule:
                continue
            if operation == "enable":
                rule.enabled = True
            elif operation == "disable":
                rule.enabled = False
            processed.append({"rule_id": rule_id, "status": "success", "new_state": "enabled" if rule.enabled else "disabled"})

        return {
            "operation_id": f"bulk_{uuid.uuid4().hex[:8]}",
            "operation": operation,
            "total_rules": len(rule_ids),
            "processed_rules": len(processed),
            "successful_rules": len(processed),
            "failed_rules": len(rule_ids) - len(processed),
            "results": processed,
            "execution_time_ms": 10.0,
            "errors": []
        }

    def export_rules(self, format_type: str = "json") -> Dict[str, Any]:
        return {
            "export_id": f"export_{uuid.uuid4().hex[:8]}",
            "format": format_type,
            "rules_exported": len(self._rules),
            "export_url": f"/exports/rules_{datetime.utcnow().date()}.{format_type}",
            "expires_at": datetime.utcnow().isoformat() + "Z",
            "checksum": uuid.uuid4().hex,
        }

    def import_rules(self, import_url: str, conflict_resolution: str = "overwrite", validate_only: bool = False) -> Dict[str, Any]:
        return {
            "import_id": f"import_{uuid.uuid4().hex[:8]}",
            "rules_imported": len(self._rules),
            "rules_updated": 0,
            "rules_created": 0,
            "validation_errors": [],
            "conflicts_resolved": 0,
            "import_status": "completed" if not validate_only else "validated",
        }

    def validate_type_specific(self, rule_type: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        requirements = {
            "validation": ["conditions", "actions"],
            "cleaning": ["conditions", "transformation_rules"],
            "transformation": ["input_fields", "output_fields"],
            "enrichment": ["source_fields", "enrichment_logic"],
        }
        return {
            "rule_type": rule_type,
            "validation_passed": True,
            "type_specific_requirements": requirements,
            "recommended_config": {},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _serialize_rule(self, rule: RuleDefinition) -> Dict[str, Any]:
        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "type": rule.type,
            "description": rule.description,
            "version": rule.version,
            "enabled": rule.enabled,
            "conditions": rule.conditions or [],
            "metadata": rule.metadata,
            "last_modified": rule.created_at.isoformat() + "Z",
            "usage_count": 0,
        }


__all__ = ["RulesService", "RuleDefinition"]
