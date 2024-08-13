import csv
import json
from jinja2 import Template

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def generate_html():
    causumx_data = read_csv('experiment_results_causumx.csv')
    greedy_data = read_csv('experiment_results_greedy.csv')

    with open('slides_template.html', 'r') as f:
        template = Template(f.read())

    slides_content = []
    for k in range(4, 8):
        causumx_row = next(row for row in causumx_data if int(row['k']) == k)
        greedy_row = next(row for row in greedy_data if int(row['k']) == k)

        causumx_rules = json.loads(causumx_row['selected_rules'])
        greedy_rules = json.loads(greedy_row['selected_rules'])

        slides_content.append({
            'k': k,
            'causumx': {
                'execution_time': float(causumx_row['execution_time']),
                'expected_utility': float(causumx_row['expected_utility']),
                'protected_expected_utility': float(causumx_row['protected_expected_utility']),
                'coverage': float(causumx_row['coverage']) * 100,
                'protected_coverage': float(causumx_row['protected_coverage']) * 100,
                'rules': causumx_rules
            },
            'greedy': {
                'execution_time': float(greedy_row['execution_time']),
                'expected_utility': float(greedy_row['expected_utility']),
                'protected_expected_utility': float(greedy_row['protected_expected_utility']),
                'coverage': float(greedy_row['coverage']) * 100,
                'protected_coverage': float(greedy_row['protected_coverage']) * 100,
                'rules': greedy_rules
            }
        })

    rendered_html = template.render(slides=slides_content)

    with open('slides.html', 'w') as f:
        f.write(rendered_html)

if __name__ == '__main__':
    generate_html()
