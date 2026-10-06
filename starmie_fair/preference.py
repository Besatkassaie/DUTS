"""
PreferenceConfig: Configurable preference relation for candidate table dominance.

Based on Chomicki (TODS 2003) preference formulas. Loads rules from a JSON
config file defining when candidate t1 dominates t2 using pairwise attribute
comparisons with arbitrary AND/OR/NOT nesting.
"""

import os
import json


class PreferenceConfig:
    """
    Configurable preference relation for candidate table dominance.

    Config format:
        {
          "preference": {
            "and": [
              {"left": "protected_count", "op": ">", "right": "protected_count"},
              {"left": "unionability_score", "op": ">=", "right": "unionability_score"}
            ]
          }
        }

    "left" refers to t1's attribute value, "right" refers to t2's attribute value.
    Supported operators: >, >=, <, <=, ==, !=
    Supported logical operators: and, or, not
    """

    _OPS = {
        '>':  lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        '<':  lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }

    def __init__(self, config_path=None):
        """
        Load preference config from JSON file.

        Args:
            config_path: Path to preference_config.json.
                         If None, uses a default rule: t1 dominates t2 if
                         protected_count(t1) > protected_count(t2) AND
                         unionability_score(t1) >= unionability_score(t2).
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.rule = config.get('preference', {})
            self.description = config.get('description', '')
        else:
            self.rule = {
                'and': [
                    {'left': 'protected_count', 'op': '>', 'right': 'protected_count'},
                    {'left': 'unionability_score', 'op': '>=', 'right': 'unionability_score'}
                ]
            }
            self.description = 'default: more protected AND >= unionability'

    def dominates(self, t1_attrs, t2_attrs):
        """
        Evaluate whether t1 dominates t2 under the configured preference relation.

        Args:
            t1_attrs: dict of attribute values for candidate t1
            t2_attrs: dict of attribute values for candidate t2

        Returns:
            True if t1 is preferred over t2
        """
        return self._evaluate(self.rule, t1_attrs, t2_attrs)

    def _evaluate(self, rule, t1, t2):
        """Recursively evaluate a preference rule node."""
        if 'and' in rule:
            return all(self._evaluate(sub, t1, t2) for sub in rule['and'])
        elif 'or' in rule:
            return any(self._evaluate(sub, t1, t2) for sub in rule['or'])
        elif 'not' in rule:
            return not self._evaluate(rule['not'], t1, t2)
        elif 'left' in rule and 'op' in rule and 'right' in rule:
            attr_left = rule['left']
            attr_right = rule['right']
            op = rule['op']
            if op not in self._OPS:
                raise ValueError(f"Unknown operator: {op}")
            val1 = t1.get(attr_left, 0)
            val2 = t2.get(attr_right, 0)
            return self._OPS[op](val1, val2)
        else:
            raise ValueError(f"Invalid preference rule node: {rule}")

    def __repr__(self):
        return f"PreferenceConfig({self.description})"
