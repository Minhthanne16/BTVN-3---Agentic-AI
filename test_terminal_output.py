import json
from pathlib import Path
import subprocess
import sys
import unittest

import hybrid_agent
import plan_execute_agent
import react_agent

PROJECT = Path(__file__).resolve().parent


class TerminalOutputTests(unittest.TestCase):
    def test_blocked_tool_has_reason_and_does_not_create_booking(self):
        result = plan_execute_agent.run(2)
        blocked = [event for event in result['trace'] if event['outcome'] == 'blocked']
        self.assertEqual(blocked[0]['tool'], 'book_seat')
        self.assertIn('Giờ bay', blocked[0]['reason'])
        self.assertIsNone(result['booking'])
        self.assertEqual(result['tool_calls'], 2)

    def test_hybrid_trace_records_recovery_to_verified_flight(self):
        result = hybrid_agent.run(2)
        self.assertEqual(result['initial_plan'][0], 'search_flights')
        self.assertEqual(result['replans'], 1)
        self.assertEqual(result['booking']['flight_id'], 'VN122')
        successful_bookings = [event for event in result['trace']
                               if event['tool'] == 'book_seat' and event['outcome'] == 'ok']
        self.assertEqual([event['args']['flight_id'] for event in successful_bookings], ['VN122'])

    def test_all_original_benchmark_outcomes_and_counters_are_preserved(self):
        benchmark = json.loads((PROJECT / 'benchmark_three_agents.json').read_text(encoding='utf-8'))
        agents = {'react': react_agent, 'plan-execute': plan_execute_agent, 'hybrid': hybrid_agent}
        for case in benchmark['cases']:
            with self.subTest(agent=case['agent'], scenario=case['scenario']):
                result = agents[case['agent']].run(case['scenario'])
                for field in ('status', 'model_calls', 'tool_calls', 'estimated_tokens',
                              'safety_catches', 'replans', 'attempts'):
                    self.assertEqual(result[field], case[field])

    def test_readable_cli_and_json_mode(self):
        pretty = subprocess.run([sys.executable, 'hybrid_agent.py', '--scenario', '2'],
                                cwd=PROJECT, capture_output=True, encoding='utf-8', check=True).stdout
        for section in ('--- plan ---', '--- recovery ---', '--- trace ---', '--- result ---'):
            self.assertIn(section, pretty)
        self.assertIn('blocked', pretty)
        raw = subprocess.run([sys.executable, 'react_agent.py', '--scenario', '1', '--json'],
                             cwd=PROJECT, capture_output=True, encoding='utf-8', check=True).stdout
        self.assertEqual(json.loads(raw)['status'], 'COMPLETED')

    def test_interactive_menu_validates_choice_and_approval(self):
        result = subprocess.run([sys.executable, 'hybrid_agent.py', '--interactive'],
                                cwd=PROJECT, input='bad\n3\ninvalid\nn\n', capture_output=True,
                                encoding='utf-8', check=True)
        self.assertIn('APPROVAL_REQUIRED', result.stdout)
        self.assertIn('Choose scenario', result.stdout)
        self.assertIn('y/n', result.stdout)

    def test_permission_and_repeated_search_are_reported_as_blocked(self):
        permission = hybrid_agent.run(3)
        self.assertEqual(permission['status'], 'APPROVAL_REQUIRED')
        self.assertEqual(permission['trace'][-1]['outcome'], 'blocked')
        self.assertIsNone(permission['booking'])
        loop = react_agent.run(4)
        self.assertEqual(loop['trace'][-1]['outcome'], 'blocked')
        self.assertIn('LOOP', loop['trace'][-1]['reason'])
        self.assertEqual(loop['tool_calls'], 1)


if __name__ == '__main__':
    unittest.main()
