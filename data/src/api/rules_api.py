"""
Rules API class for preprocessing rule management.

This module provides API endpoints for creating, managing, and executing
data preprocessing rules including validation, cleaning, and transformation rules.
"""

import json
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
import re


class RuleType(Enum):
    """Rule type enumeration."""
    VALIDATION = "validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"


class RuleStatus(Enum):
    """Rule status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"


@dataclass
class Rule:
    """Rule data structure."""
    id: str
    name: str
    description: str
    rule_type: RuleType
    status: RuleStatus
    version: str
    parameters: Dict[str, Any]
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: Optional[str] = None


@dataclass
class RuleExecution:
    """Rule execution result."""
    rule_id: str
    execution_id: str
    status: str
    data_points_processed: int
    violations_found: int
    actions_taken: List[Dict[str, Any]]
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None


class RulesAPI:
    """API class for preprocessing rule management."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize RulesAPI with database path."""
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '..', '..', 'storage', 'preprocessing_rules.db')
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for rules management."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    rule_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT
                )
            ''')

            # Create rule_executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rule_executions (
                    execution_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_points_processed INTEGER DEFAULT 0,
                    violations_found INTEGER DEFAULT 0,
                    actions_taken TEXT,
                    execution_time REAL DEFAULT 0.0,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (rule_id) REFERENCES rules (id)
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_type ON rules(rule_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_status ON rules(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_executions_rule_id ON rule_executions(rule_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_executions_timestamp ON rule_executions(timestamp)')

            conn.commit()

    def create_rule_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new preprocessing rule.

        Args:
            request_data: Dictionary containing rule definition

        Returns:
            Dictionary with creation result
        """
        try:
            # Validate required fields
            required_fields = ['name', 'rule_type', 'parameters', 'conditions', 'actions']
            for field in required_fields:
                if field not in request_data:
                    return {
                        'status': 'error',
                        'message': f'Missing required field: {field}',
                        'code': 400
                    }

            # Validate rule type
            try:
                rule_type = RuleType(request_data['rule_type'])
            except ValueError:
                return {
                    'status': 'error',
                    'message': f'Invalid rule_type: {request_data["rule_type"]}',
                    'code': 400
                }

            # Validate rule structure
            validation_result = self._validate_rule_structure(request_data)
            if validation_result['status'] == 'error':
                return validation_result

            # Create rule object
            rule = Rule(
                id=str(uuid.uuid4()),
                name=request_data['name'],
                description=request_data.get('description', ''),
                rule_type=rule_type,
                status=RuleStatus.ACTIVE,
                version="1.0.0",
                parameters=request_data['parameters'],
                conditions=request_data['conditions'],
                actions=request_data['actions'],
                metadata=request_data.get('metadata'),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                created_by=request_data.get('created_by')
            )

            # Store rule in database
            self._store_rule(rule)

            return {
                'status': 'success',
                'data': {
                    'rule_id': rule.id,
                    'name': rule.name,
                    'rule_type': rule.rule_type.value,
                    'status': rule.status.value,
                    'version': rule.version,
                    'created_at': rule.created_at
                },
                'metadata': {
                    'operation': 'create_rule',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rule creation failed: {str(e)}',
                'code': 500
            }

    def _validate_rule_structure(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rule structure and parameters."""
        try:
            # Validate conditions
            conditions = rule_data.get('conditions', [])
            for condition in conditions:
                if not isinstance(condition, dict):
                    return {
                        'status': 'error',
                        'message': 'Conditions must be dictionaries',
                        'code': 400
                    }

                if 'field' not in condition or 'operator' not in condition:
                    return {
                        'status': 'error',
                        'message': 'Conditions must contain "field" and "operator"',
                        'code': 400
                    }

                # Validate operator
                valid_operators = ['equals', 'not_equals', 'greater_than', 'less_than', 'contains', 'regex']
                if condition['operator'] not in valid_operators:
                    return {
                        'status': 'error',
                        'message': f'Invalid operator: {condition["operator"]}',
                        'code': 400
                    }

            # Validate actions
            actions = rule_data.get('actions', [])
            for action in actions:
                if not isinstance(action, dict):
                    return {
                        'status': 'error',
                        'message': 'Actions must be dictionaries',
                        'code': 400
                    }

                if 'type' not in action:
                    return {
                        'status': 'error',
                        'message': 'Actions must contain "type"',
                        'code': 400
                    }

                # Validate action types based on rule type
                rule_type = rule_data.get('rule_type')
                valid_actions = self._get_valid_actions_for_rule_type(rule_type)
                if action['type'] not in valid_actions:
                    return {
                        'status': 'error',
                        'message': f'Invalid action type {action["type"]} for rule type {rule_type}',
                        'code': 400
                    }

            return {
                'status': 'success',
                'message': 'Rule structure is valid'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rule validation failed: {str(e)}',
                'code': 500
            }

    def _get_valid_actions_for_rule_type(self, rule_type: str) -> List[str]:
        """Get valid action types for a given rule type."""
        action_mapping = {
            'validation': ['flag', 'reject', 'warn', 'log'],
            'cleaning': ['remove', 'replace', 'interpolate', 'clip'],
            'transformation': ['normalize', 'scale', 'encode', 'aggregate'],
            'enrichment': ['calculate', 'derive', 'join', 'lookup']
        }
        return action_mapping.get(rule_type, [])

    def _store_rule(self, rule: Rule):
        """Store rule in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO rules (
                    id, name, description, rule_type, status, version,
                    parameters, conditions, actions, metadata, created_at, updated_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.id,
                rule.name,
                rule.description,
                rule.rule_type.value,
                rule.status.value,
                rule.version,
                json.dumps(rule.parameters),
                json.dumps(rule.conditions),
                json.dumps(rule.actions),
                json.dumps(rule.metadata),
                rule.created_at,
                rule.updated_at,
                rule.created_by
            ))
            conn.commit()

    def get_rules_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get list of rules with optional filtering.

        Args:
            request_data: Dictionary containing filter criteria

        Returns:
            Dictionary with rules list
        """
        try:
            # Build query parameters
            rule_type = request_data.get('rule_type')
            status = request_data.get('status')
            limit = request_data.get('limit', 100)

            # Query rules from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM rules WHERE 1=1"
                params = []

                if rule_type:
                    query += " AND rule_type = ?"
                    params.append(rule_type)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

            # Convert to rule objects
            rules = []
            for row in rows:
                rule = Rule(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    rule_type=RuleType(row[3]),
                    status=RuleStatus(row[4]),
                    version=row[5],
                    parameters=json.loads(row[6]),
                    conditions=json.loads(row[7]),
                    actions=json.loads(row[8]),
                    metadata=json.loads(row[9]) if row[9] else None,
                    created_at=row[10],
                    updated_at=row[11],
                    created_by=row[12]
                )
                rules.append(rule)

            return {
                'status': 'success',
                'data': {
                    'rules': [asdict(rule) for rule in rules],
                    'total_count': len(rules),
                    'filters_applied': {
                        'rule_type': rule_type,
                        'status': status,
                        'limit': limit
                    }
                },
                'metadata': {
                    'query_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rules retrieval failed: {str(e)}',
                'code': 500
            }

    def get_rule_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a specific rule by ID.

        Args:
            request_data: Dictionary containing rule_id

        Returns:
            Dictionary with rule details
        """
        try:
            rule_id = request_data.get('rule_id')
            if not rule_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: rule_id',
                    'code': 400
                }

            # Query rule from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM rules WHERE id = ?", (rule_id,))
                row = cursor.fetchone()

            if not row:
                return {
                    'status': 'error',
                    'message': f'Rule not found: {rule_id}',
                    'code': 404
                }

            # Convert to rule object
            rule = Rule(
                id=row[0],
                name=row[1],
                description=row[2],
                rule_type=RuleType(row[3]),
                status=RuleStatus(row[4]),
                version=row[5],
                parameters=json.loads(row[6]),
                conditions=json.loads(row[7]),
                actions=json.loads(row[8]),
                metadata=json.loads(row[9]) if row[9] else None,
                created_at=row[10],
                updated_at=row[11],
                created_by=row[12]
            )

            return {
                'status': 'success',
                'data': asdict(rule),
                'metadata': {
                    'query_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rule retrieval failed: {str(e)}',
                'code': 500
            }

    def update_rule_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing rule.

        Args:
            request_data: Dictionary containing rule updates

        Returns:
            Dictionary with update result
        """
        try:
            rule_id = request_data.get('rule_id')
            if not rule_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: rule_id',
                    'code': 400
                }

            # Check if rule exists
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM rules WHERE id = ?", (rule_id,))
                existing_rule = cursor.fetchone()

            if not existing_rule:
                return {
                    'status': 'error',
                    'message': f'Rule not found: {rule_id}',
                    'code': 404
                }

            # Validate updates
            validation_result = self._validate_rule_structure(request_data)
            if validation_result['status'] == 'error':
                return validation_result

            # Update rule
            update_fields = []
            update_params = []

            for field in ['name', 'description', 'rule_type', 'status', 'parameters', 'conditions', 'actions', 'metadata']:
                if field in request_data:
                    update_fields.append(f"{field} = ?")
                    if field in ['parameters', 'conditions', 'actions', 'metadata']:
                        update_params.append(json.dumps(request_data[field]))
                    else:
                        update_params.append(request_data[field])

            if update_fields:
                update_fields.append("updated_at = ?")
                update_params.append(datetime.now().isoformat())
                update_params.append(rule_id)

                query = f"UPDATE rules SET {', '.join(update_fields)} WHERE id = ?"

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, update_params)
                    conn.commit()

            return {
                'status': 'success',
                'data': {
                    'rule_id': rule_id,
                    'updated_fields': [field.split(' =')[0] for field in update_fields[:-2]],  # Exclude updated_at and id
                    'updated_at': datetime.now().isoformat()
                },
                'metadata': {
                    'operation': 'update_rule',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rule update failed: {str(e)}',
                'code': 500
            }

    def delete_rule_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete a rule.

        Args:
            request_data: Dictionary containing rule_id

        Returns:
            Dictionary with deletion result
        """
        try:
            rule_id = request_data.get('rule_id')
            if not rule_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: rule_id',
                    'code': 400
                }

            # Check if rule exists
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM rules WHERE id = ?", (rule_id,))
                existing_rule = cursor.fetchone()

            if not existing_rule:
                return {
                    'status': 'error',
                    'message': f'Rule not found: {rule_id}',
                    'code': 404
                }

            # Delete rule
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
                conn.commit()

            return {
                'status': 'success',
                'data': {
                    'rule_id': rule_id,
                    'deleted_at': datetime.now().isoformat()
                },
                'metadata': {
                    'operation': 'delete_rule',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rule deletion failed: {str(e)}',
                'code': 500
            }

    def test_rule_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a rule against sample data.

        Args:
            request_data: Dictionary containing rule_id and test_data

        Returns:
            Dictionary with test results
        """
        try:
            rule_id = request_data.get('rule_id')
            test_data = request_data.get('test_data')

            if not rule_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: rule_id',
                    'code': 400
                }

            if not test_data:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: test_data',
                    'code': 400
                }

            # Get rule
            rule_result = self.get_rule_endpoint({'rule_id': rule_id})
            if rule_result['status'] == 'error':
                return rule_result

            rule_data = rule_result['data']

            # Simulate rule execution
            execution_result = self._simulate_rule_execution(rule_data, test_data)

            return {
                'status': 'success',
                'data': execution_result,
                'metadata': {
                    'operation': 'test_rule',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Rule test failed: {str(e)}',
                'code': 500
            }

    def _simulate_rule_execution(self, rule: Dict[str, Any], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate rule execution against test data."""
        start_time = datetime.now()

        try:
            # This is a simplified simulation
            # In a real implementation, you would execute the actual rule logic

            conditions = rule['conditions']
            actions = rule['actions']
            rule_type = rule['rule_type']

            violations_found = 0
            actions_taken = []

            # Simulate checking conditions
            for condition in conditions:
                field = condition.get('field')
                operator = condition.get('operator')
                value = condition.get('value')

                # Simple simulation - count potential violations
                if field in test_data:
                    data_values = test_data[field] if isinstance(test_data[field], list) else [test_data[field]]
                    violations_found += len(data_values)  # Simplified count

            # Simulate taking actions
            for action in actions:
                action_type = action.get('type')
                actions_taken.append({
                    'type': action_type,
                    'applied': True,
                    'items_affected': violations_found
                })

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                'rule_id': rule['id'],
                'execution_id': str(uuid.uuid4()),
                'status': 'completed',
                'data_points_processed': len(test_data),
                'violations_found': violations_found,
                'actions_taken': actions_taken,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'rule_type': rule_type,
                'simulation': True
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'rule_id': rule['id'],
                'execution_id': str(uuid.uuid4()),
                'status': 'failed',
                'data_points_processed': 0,
                'violations_found': 0,
                'actions_taken': [],
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'error_message': str(e),
                'simulation': True
            }

    def bulk_operations_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform bulk operations on rules.

        Args:
            request_data: Dictionary containing bulk operation details

        Returns:
            Dictionary with bulk operation results
        """
        try:
            operation = request_data.get('operation')
            rule_ids = request_data.get('rule_ids', [])

            if not operation:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: operation',
                    'code': 400
                }

            if not rule_ids:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: rule_ids',
                    'code': 400
                }

            valid_operations = ['activate', 'deactivate', 'delete']
            if operation not in valid_operations:
                return {
                    'status': 'error',
                    'message': f'Invalid operation: {operation}. Valid operations: {valid_operations}',
                    'code': 400
                }

            results = []
            for rule_id in rule_ids:
                try:
                    if operation == 'activate':
                        result = self.update_rule_endpoint({
                            'rule_id': rule_id,
                            'status': 'active'
                        })
                    elif operation == 'deactivate':
                        result = self.update_rule_endpoint({
                            'rule_id': rule_id,
                            'status': 'inactive'
                        })
                    elif operation == 'delete':
                        result = self.delete_rule_endpoint({'rule_id': rule_id})

                    results.append({
                        'rule_id': rule_id,
                        'status': 'success' if result['status'] == 'success' else 'failed',
                        'message': result.get('message', '')
                    })

                except Exception as e:
                    results.append({
                        'rule_id': rule_id,
                        'status': 'failed',
                        'message': str(e)
                    })

            success_count = sum(1 for r in results if r['status'] == 'success')
            total_count = len(results)

            return {
                'status': 'success',
                'data': {
                    'operation': operation,
                    'total_rules': total_count,
                    'successful': success_count,
                    'failed': total_count - success_count,
                    'results': results
                },
                'metadata': {
                    'operation': 'bulk_operations',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Bulk operation failed: {str(e)}',
                'code': 500
            }