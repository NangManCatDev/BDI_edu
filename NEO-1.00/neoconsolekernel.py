from ipykernel.kernelbase import Kernel
from ctypes import *
import os
import pandas as pd
import re
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import random  # 무작위 위치 할당을 위해 사용
import matplotlib.pyplot as plt  # 그래프 그리기를 위해 사용
import datetime
import pathlib

# 전역 변수 초기화
agent_list = {}
var_return = {}
model_config = {}
temp = []
var_to_check = {}
final_return = []
agent_to_remove = []
tik = 0
instance_size = 0
agent_to_add = 0
class_name = ""

def gini_index(values):
    """
    지니 계수를 계산하는 함수
    todo:
    에이전트 타입별로 계산할지 등등 구현
    """
    n = len(values)
    if n == 0:
        return 0

    values.sort()
    sum_of_values = sum(values)
    if sum_of_values == 0:
        return 0

    gini = 1
    for i, value in enumerate(values, start=1):
        gini -= (2 * i - n - 1) * value / (n * sum_of_values)

    return gini

import numpy as np

# 에이전트의 속성 값이 wealth면 정규분포 시키는 함수
def assign_normal_distribution(agent_type, variable_name, mean=10, std_dev=1):
    """
    주어진 agent_type에 대해 variable_name이 'wealth'일 경우 
    정규분포를 따라 값을 생성하여 각 에이전트에 할당한다.
    """
    global var_return, agent_list
    
    if variable_name == "wealth" and agent_type in agent_list:
        # 에이전트 수 가져오기
        num_agents = len(agent_list[agent_type])
        
        # 정규분포를 사용해 값 생성
        wealth_values = np.random.normal(mean, std_dev, num_agents)
        
        # 결과를 var_return에 저장
        if agent_type not in var_return:
            var_return[agent_type] = {}
        
        var_return[agent_type][variable_name] = wealth_values.tolist()
        
        # 각 에이전트에 값 할당 (가상 처리)
        for idx, agent in enumerate(agent_list[agent_type]):
            print(f"Assigned wealth to {agent}: {wealth_values[idx]}")  # 확인용 출력


class NeoConsoleKernel(Kernel):
    implementation = 'Neo Console'
    implementation_version = '1.1'
    language = 'C'
    language_version = '3.7'
    language_info = {'name': 'NeoConsole',
                     'mimetype': 'text/plain',
                     'extension': '.py'}
    banner = "Kernel for Neo Console App"

    def __init__(self, **kwargs):
        super(NeoConsoleKernel, self).__init__(**kwargs)
        str_tgt = os.path.abspath("libNeoDLL.so")
        print(str_tgt)
        self.neodll = cdll.LoadLibrary(str_tgt)
        self.neoInit = self.neodll.NEO_Init
        self.neoExit = self.neodll.NEO_Exit
        self.neoEventEngine = self.neodll.NEO_EventEngine
        self.neoEventEngine.argtypes = [c_char_p, c_char_p]
        self.neoEventEngine.restype = c_int
        self.neoInit()
        
        # 타임스탬프를 사용하여 출력 디렉토리 생성
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = pathlib.Path(f'output_{timestamp}')
        self.output_dir.mkdir(exist_ok=True)

    def _write_to_log(self, code):
        """
        입력된 코드를 로그 파일에 작성하는 도우미 함수
        """
        log_path = self.output_dir / 'log.txt'
        with open(log_path, 'a') as log_file:
            log_file.write(code + '\n' + '-'*40 + '\n')

    def _execute_command(self, line, do_print=True):
        """
        명령을 실행하고 결과를 반환하는 함수
        """
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        line = line.replace(" "," ")
        
        szCommandBuf = line.encode('utf-8')
        szOutput = create_string_buffer(1000)
        self.neoEventEngine(szCommandBuf, szOutput)
        response = szOutput.value.decode('utf-8').strip()
        sys.stdout = old_stdout
        stdout = mystdout.getvalue()

        try:
            with open('output.txt', 'r') as file:
                output_contents = file.read()
        except FileNotFoundError:
            output_contents = 'output.txt not found or unable to read.'
        if do_print:
            stream_content = {'name': 'stdout', 'text': output_contents + '\n' + stdout + 'result: ' + response}
            self.send_response(self.iopub_socket, 'stream', stream_content)

        return response

    def _handle_agent_creation(self, line):
        """
        에이전트 생성 처리 함수
        """
        match = re.search(r'is-a (\w+)', line)
        if match:
            agent_type = match.group(1)
            if agent_type not in agent_list:
                agent_list[agent_type] = []

            line = line.replace(f"is-a {agent_type}", "is-a prototype")
            self._execute_command(line, do_print=False)
            line = line.replace("is-a prototype", f"type {agent_type}")
            self._execute_command(line)

    def _handle_grid_creation(self, line):
        """
        그리드 생성 처리 함수
        todo:
        지역을 어떻게 반영할지 고려.. 황기우 선생님 원하는 방향대로 될 수 있도록 구현
        """
        line = line.replace("is-a grid", "is-a prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type grid")
        self._execute_command(line)

    def _handle_map_creation(self, line):
        """
        맵 생성 처리 함수
        todo:
        지역을 어떻게 반영할지 고려.. 황기우 선생님 원하는 방향대로 될 수 있도록 구현
        """
        line = line.replace("is-a map", "is-a prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type map")
        self._execute_command(line)

    def _handle_rule_creation(self, line):
        """
        규칙 생성 처리 함수
        """
        line = line.replace("is-a rule", "is-a prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type rule")
        self._execute_command(line)

    def _handle_agent_sim_model_creation(self, line):
        """
        에이전트 시뮬레이션 모델 생성 처리 함수
        """
        line = line.replace("is-a AgentSimModel", "is-a prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type AgentSimModel")
        self._execute_command(line)

    def _handle_region_creation(self, line):
        """
        지역 생성 처리 함수
        todo:
        지역을 어떻게 반영할지 고려.. 황기우 선생님 원하는 방향대로 될 수 있도록 구현
        """
        line = line.replace("is-a region", "is-a prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type region")
        self._execute_command(line)

    def _handle_sim_start(self, line):
        """
        시뮬레이션 시작 처리 함수
        """
        start_time = time.time()
        
        create_only_agent = False
        if " only-create-agent" in line:
            create_only_agent = True
            line = line.replace(" only-create-agent", "")
        multi_agent_group = False
        multi_class_group = False

        # 시뮬레이션 설정 가져오기
        simulation_name = re.search(r'\(sim\s+(\S+)\)', line).group(1)
        line = "(get-list '" + simulation_name + " 'name)"
        response = self._execute_command(line, do_print=False)
        model_config['name'] = str(response).strip('"')

        line = "(get-list '" + simulation_name + " 'simulation-start)"
        response = self._execute_command(line, do_print=False)
        model_config['start'] = int(str(response))

        line = "(get-list '" + simulation_name + " 'simulation-end)"
        response = self._execute_command(line, do_print=False)
        model_config['end'] = int(str(response))

        line = "(get-list '" + simulation_name + " 'tik-rule)"
        response = self._execute_command(line, do_print=False)
        model_config['tik-rule'] = str(response).strip('"')

        line = "(get-list '" + simulation_name + " 'result-csv-file)"
        response = self._execute_command(line, do_print=False)
        model_config['result-file'] = str(response).strip('"')

        line = "(get-list '" + simulation_name + " 'primitive-variables)"
        response = self._execute_command(line, do_print=False)
        model_config['primitive-variables'] = str(response)

        line = "(get-list '" + simulation_name + " 'calculated-variables)"
        response = self._execute_command(line, do_print=False)
        model_config['calculated-variables'] = str(response).replace("(","").replace(")","").split(" ")

        line = "(get-list '" + simulation_name + " 'agent-groups)"
        response = self._execute_command(line, do_print=False)
        model_config['agent-groups'] = str(response).replace("(","").replace(")","")

        if len(model_config['agent-groups'].split(" ")) > 1:
            multi_agent_group = True
            model_config['agent-type'] = ""
            agent_group_names = model_config['agent-groups'].split(" ")
            for agent_group_name in agent_group_names:
                line = "(get-list '" + agent_group_name + " 'agent-type)"
                response = self._execute_command(line, do_print=False)
                model_config['agent-type'] = model_config['agent-type'] + " " + str(response)
            model_config['agent-type'] = model_config['agent-type'][1:]

            if len(set(model_config['agent-type'].split(" "))) > 1:
                multi_class_group = True            

        else:
            agent_group_name = model_config['agent-groups']
            line = "(get-list '" + agent_group_name + " 'agent-type)"
            response = self._execute_command(line, do_print=False)
            model_config['agent-type'] = str(response)

        if multi_agent_group:
            model_config['agent-size'] = ""
            agent_group_names = model_config['agent-groups'].split(" ")
            for agent_group_name in agent_group_names:
                line = "(get-list '" + agent_group_name + " 'size)"
                response = self._execute_command(line, do_print=False)
                model_config['agent-size'] = model_config['agent-size'] + " " + str(response)
            model_config['agent-size'] = model_config['agent-size'][1:]
        else:
            line = "(get-list '" + agent_group_name + " 'size)"
            response = self._execute_command(line, do_print=False)
            model_config['agent-size'] = str(response)

        if multi_class_group:
            model_config['agent-tik-rules'] = ""
            types = model_config['agent-type'].split(" ")
            agent_group_names = model_config['agent-groups'].split(" ")
            for agent_type in types:
                line = "(get-list '" + agent_type + " 'tik-rule)"
                response = self._execute_command(line, do_print=False)
                model_config['agent-tik-rules'] = model_config['agent-tik-rules'] + " " + str(response)
            model_config['agent-tik-rules'] = model_config['agent-tik-rules'][1:]
        else:    
            line = "(get-list '" + model_config['agent-type'].split(" ")[0] + " 'tik-rule)"
            response = self._execute_command(line, do_print=False)
            model_config['agent-tik-rules'] = str(response)

        # 그리드 크기와 지역 스타일 추출
        # todo:
        # 지역을 어떻게 반영할지 고려.. 황기우 선생님 원하는 방향대로 될 수 있도록 구현
        cmd = "(get-list 'grid1 'size)"
        grid_size_response = self._execute_command(cmd, do_print=False)
        grid_size = tuple(map(int, grid_size_response.strip('()').split()))
        cmd = "(get-list 'region1 'region-style)"
        region_style_response = self._execute_command(cmd, do_print=False).strip('"')

        # 에이전트 생성
        if multi_agent_group:
            if multi_class_group:
                for ii in range(len(model_config['agent-groups'].split(" "))):
                    class_name = model_config['agent-type'].split(" ")[ii]
                    agent_list[class_name] = []

                    instance_name = model_config['agent-groups'].split(" ")[ii]

                    instance_size = int(model_config['agent-size'].split(" ")[ii])
                    self._write_to_log(str(instance_size))
                    temp_names = ""
                    for i in range(1, instance_size+1):
                        self._write_to_log(str(i))
                        new_name = class_name + "-" + instance_name + "-" + str(i)
                        agent_list[class_name].append(new_name)
                        x_pos = random.randint(0, grid_size[0] - 1)
                        y_pos = random.randint(0, grid_size[1] - 1)
                        if i != (instance_size+1):
                            temp_names = temp_names + new_name + " "
                        else:
                            temp_names = temp_names + new_name
                        line = f"(keep '({new_name} is-a {class_name} x {x_pos} y {y_pos} region {region_style_response}))"
                        self._execute_command(line, do_print=False)
                    line = "(keep '(" + instance_name + " list (" + temp_names + ")))"
                    self._execute_command(line)
                    response = "instance created"
                    try:
                        with open('output.txt', 'r') as file:
                            output_contents = file.read()
                    except FileNotFoundError:
                        output_contents = 'output.txt not found or unable to read.'
                    stream_content = {'name': 'stdout', 'text': output_contents + '\n' + 'result: ' + response}
                    self.send_response(self.iopub_socket, 'stream', stream_content)
            else:
                class_name = model_config['agent-type'].split(" ")[0]
                agent_list[class_name] = []

                for ii in range(len(agent_group_names)):
                    time.sleep(0.5)
                    instance_name = model_config['agent-groups'].split(" ")[ii]

                    instance_size = int(model_config['agent-size'].split(" ")[ii])
                    self._write_to_log(str(instance_size))
                    temp_names = ""
                    for i in range(1, instance_size+1):
                        self._write_to_log(str(i))
                        new_name = class_name + "-" + instance_name + "-" + str(i)
                        agent_list[class_name].append(new_name)
                        x_pos = random.randint(0, grid_size[0] - 1)
                        y_pos = random.randint(0, grid_size[1] - 1)
                        if i != (instance_size+1):
                            temp_names = temp_names + new_name + " "
                        else:
                            temp_names = temp_names + new_name
                        line = f"(keep '({new_name} is-a {class_name} x {x_pos} y {y_pos} region {region_style_response}))"
                        self._execute_command(line, do_print=False)
                    line = "(keep '(" + instance_name + " list (" + temp_names + ")))"
                    self._execute_command(line)
                    response = "instance created"
                    try:
                        with open('output.txt', 'r') as file:
                            output_contents = file.read()
                    except FileNotFoundError:
                        output_contents = 'output.txt not found or unable to read.'
                    stream_content = {'name': 'stdout', 'text': output_contents + '\n' + 'result: ' + response}
                    self.send_response(self.iopub_socket, 'stream', stream_content)
        else:
            class_name = model_config['agent-type']
            instance_name = model_config['agent-groups']
            agent_list[class_name] = []

            instance_size = int(model_config['agent-size'])
            self._write_to_log(str(instance_size))
            temp_names = ""
            for i in range(1, instance_size+1):
                self._write_to_log(str(i))
                new_name = class_name + "-" + instance_name + "-" + str(i)
                agent_list[class_name].append(new_name)
                x_pos = random.randint(0, grid_size[0] - 1)
                y_pos = random.randint(0, grid_size[1] - 1)
                if i != (instance_size+1):
                    temp_names = temp_names + new_name + " "
                else:
                    temp_names = temp_names + new_name
                line = f"(keep '({new_name} is-a {class_name} x {x_pos} y {y_pos} region {region_style_response}))"
                self._execute_command(line, do_print=False)
            line = "(keep '(" + instance_name + " list (" + temp_names + ")))"
            self._execute_command(line)
            response = "instance created"
            try:
                with open('output.txt', 'r') as file:
                    output_contents = file.read()
            except FileNotFoundError:
                output_contents = 'output.txt not found or unable to read.'
            stream_content = {'name': 'stdout', 'text': output_contents + '\n' + 'result: ' + response}
            self.send_response(self.iopub_socket, 'stream', stream_content)

        # 규칙 처리
        cond_list = []
        for string in temp:
            if " if " in string:
                if 'keep' in string:
                    string = string.replace("(keep '(", '')
                    string = string[:-2]  # 끝의 "))" 제거
                else:
                    string = string[1:-1]

                cond, action = string.split(' if ')

                if cond not in model_config:
                    model_config[cond] = {}

                model_config[cond]['if'] = action

            elif " then " in string:
                if 'keep' in string:
                    string = string.replace("(keep '(", '')
                    string = string[:-2]  # 끝의 "))" 제거
                else:
                    string = string[1:-1]

                cond, action = string.split(' then ')
                cond_list.append(cond)
                if cond not in model_config:
                    model_config[cond] = {}

                model_config[cond]['then'] = action

        for items in cond_list:
            line = "(get-list 'cond-name 'owned-by)"
            line = line.replace("cond-name", items)
            response = self._execute_command(line, do_print=False)
            model_config[items]['type'] = str(response)

            line = "(get-list 'cond-name 'rule-time-type)"
            line = line.replace("cond-name", items)
            response = self._execute_command(line, do_print=False)
            model_config[items]['tik'] = str(response)

        var_to_check_temp = model_config['primitive-variables'].split(") (")

        for var in var_to_check_temp:
            var = var.replace("(", "").replace(")", "")
            agent_type, variable_name = var.split(" ")

            if agent_type in var_to_check:
                var_to_check[agent_type].append(variable_name)
            else:
                var_to_check[agent_type] = [variable_name]

            # wealth인 경우 정규분포로 값 생성
            if variable_name == "wealth":
                assign_normal_distribution(agent_type, variable_name)


        class_name = model_config['agent-type'].split(" ")[0]
        rules = model_config['agent-tik-rules'].replace("(","").replace(")","").split(" ")

        iter_count = model_config['end'] - model_config['start']

        if create_only_agent:
            return

        # 시뮬레이션 루프 시작
        for simul in range(0, iter_count):
            # 반복 인덱스를 출력
            stream_content = {'name': 'stdout', 'text': f"\nIteration {simul + model_config['start']}:\n"}
            self.send_response(self.iopub_socket, 'stream', stream_content)
            
            agent_to_remove = []
            agent_to_add = 0
            total_agents = sum(len(agent_list[atype]) for atype in agent_list)
            if total_agents <= 2:
                break

            self._write_to_log("iter " + str(simul))
            line = f"(forward '{model_config['tik-rule']})"
            self._execute_command(line, do_print=False)

            for iii in range(len(set(model_config['agent-type'].split(" ")))):
                agent_type = model_config['agent-type'].split(" ")[iii]
                for agent in agent_list[agent_type]:
                    line = "(get-list self 'message-box)"
                    line = line.replace("self", "'" + agent)
                    response = self._execute_command(line, do_print=False)
                    self._write_to_log(response)
                    for rule in rules:
                        self._write_to_log(rule)
                        rule_cond = model_config.get(rule, {}).get('if', '')
                        rule_then = model_config.get(rule, {}).get('then', '')

                        if "(win)" in rule_cond:
                            if "win" in response:
                                if rule_then == "(remove-agent self)":
                                    agent_to_remove.append(agent)
                                elif rule_then == "(add-agent ag1)":
                                    agent_to_add += 1
                                else:
                                    line = rule_then
                                    line = line.replace("self", "'" + agent)
                                    self._execute_command(line, do_print=False)
                        elif "(lose)" in rule_cond:
                            if "lose" in response:
                                if rule_then == "(remove-agent self)":
                                    agent_to_remove.append(agent)
                                elif rule_then == "(add-agent ag1)":
                                    agent_to_add += 1
                                else:
                                    line = rule_then
                                    line = line.replace("self", "'" + agent)
                                    self._execute_command(line, do_print=False)
                        elif "binaryrand" in rule_cond:
                            match = re.search(r'\(binaryrand\s+(\d+\.\d+)\)', rule_cond)
                            extracted_value = float(match.group(1))
                            random_binomial = np.random.binomial(1, extracted_value)
                            if random_binomial == 1:
                                if rule_then == "(remove-agent self)":
                                    agent_to_remove.append(agent)
                                elif rule_then == "(add-agent ag1)":
                                    agent_to_add += 1
                                else:
                                    line = rule_then
                                    line = line.replace("self", "'" + agent)
                                    self._execute_command(line, do_print=False)
                        elif "(true)" in rule_cond:
                            if rule_then == "(remove-agent self)":
                                agent_to_remove.append(agent)
                            elif rule_then == "(add-agent ag1)":
                                agent_to_add += 1
                            else:
                                line = rule_then
                                line = line.replace("self", "'" + agent)
                                self._execute_command(line, do_print=False)
                        else:
                            line = rule_cond
                            line = line.replace("self", "'" + agent)
                            response = self._execute_command(line, do_print=False)
                            self._write_to_log(response)
                            if response == "t":
                                if rule_then == "(remove-agent self)":
                                    agent_to_remove.append(agent)
                                elif rule_then == "(add-agent ag1)":
                                    agent_to_add += 1
                                else:
                                    line = rule_then
                                    line = line.replace("self", "'" + agent)
                                    self._execute_command(line, do_print=False)
                            else:
                                pass

            # 원시 변수 출력
            output_text = "\nPrimitive Variables:\n"
            for agent_type in var_to_check:
                variables = var_to_check[agent_type]
                for variable_name in variables:
                    line = f"(get-variable {agent_type} {variable_name})"
                    self._handle_get_variable(line)
                    # 값 형식 지정
                    if agent_type in var_return and variable_name in var_return[agent_type]:
                        values = var_return[agent_type][variable_name]
                        output_text += f"  {agent_type}.{variable_name}: {values}\n"
            
            stream_content = {'name': 'stdout', 'text': output_text}
            self.send_response(self.iopub_socket, 'stream', stream_content)

            # 계산된 변수 출력
            output_text = "\nCalculated Variables:\n"
            for variable_name in model_config['calculated-variables']:
                if variable_name == "gini":
                    wealth_values = []
                    for agent_type in var_return:
                        if 'wealth' in var_return[agent_type]:
                            wealth_values.extend([float(w) for w in var_return[agent_type]['wealth']])
                    gini_value = gini_index(wealth_values)
                    output_text += f"  gini: {gini_value}\n"
                    final_return.append({'tik': simul + model_config['start'], 'gini': gini_value})
                if variable_name == "no_of_agent":
                    total_agents = sum(len(agent_list[atype]) for atype in agent_list)
                    output_text += f"  no_of_agent: {total_agents}\n"
                    final_return.append({'tik': simul + model_config['start'], 'no_of_agent': total_agents})
            
            stream_content = {'name': 'stdout', 'text': output_text}
            self.send_response(self.iopub_socket, 'stream', stream_content)

            # 에이전트 추가 처리
            if agent_to_add > 0:
                for agent in range(agent_to_add):
                    agent_num = agent + 1 + instance_size
                    new_name = class_name + "-" + instance_name + "-" + str(agent_num)
                    agent_list[class_name].append(new_name)
                    x_pos = random.randint(0, grid_size[0] - 1)
                    y_pos = random.randint(0, grid_size[1] - 1)
                    line = f"(keep '({new_name} is-a {class_name} x {x_pos} y {y_pos} region {region_style_response}))"
                    self._execute_command(line, do_print=False)
                instance_size += agent_to_add

            # 에이전트 제거 처리
            if len(agent_to_remove) > 0:
                self._write_to_log(' '.join(i for i in agent_to_remove))
                for agent in agent_to_remove:
                    agent_type = agent.split('-')[0]
                    ls = agent_list[agent_type]
                    if agent in ls:
                        ls.remove(agent)
                        self._write_to_log(' '.join(i for i in ls))
                        agent_list[agent_type] = ls
            else:
                for agent_type in agent_list:
                    ls = agent_list[agent_type]
                    self._write_to_log(' '.join(i for i in ls))
                    agent_list[agent_type] = ls

            # 에이전트 그룹 업데이트
            for agent_group_name in agent_group_names:
                agents_in_group = []
                for agent_type in agent_list:
                    agents_in_group.extend([agent for agent in agent_list[agent_type] if agent.startswith(f"{agent_type}-{agent_group_name}-")])
                line = "(put-list {0} 'list '({1}))".format(agent_group_name, ' '.join(agents_in_group))
                self._execute_command(line)

        end_time = time.time()
        execution_time = end_time - start_time
        
        # 완료 메시지 출력
        completion_msg = f"\nSimulation Completed in {execution_time:.2f} seconds\n"
        stream_content = {'name': 'stdout', 'text': completion_msg}
        self.send_response(self.iopub_socket, 'stream', stream_content)

        # CSV를 타임스탬프 디렉토리에 저장
        filename = self.output_dir / model_config['result-file']
        pd.DataFrame(final_return).to_csv(filename, index=False)

        # 그래프 설정 가져오기
        line = "(get-list 'graph01 'title)"
        response = self._execute_command(line, do_print=False)
        plt_title = str(response).strip('"')

        line = "(get-list 'graph01 'x-axis-label)"
        response = self._execute_command(line, do_print=False)
        xlabel = str(response).strip('"')

        line = "(get-list 'graph01 'y-axis-label)"
        response = self._execute_command(line, do_print=False)
        ylabel = str(response).strip('"')

        line = "(get-list 'graph01 'graph-type)"
        response = self._execute_command(line, do_print=False)
        graph_type = str(response).strip('"')

        # 그래프를 그리기 위한 데이터 준비
        tik_values = [item['tik'] for item in final_return]
        gini_values = [item.get('gini', 0) for item in final_return]

        plt.figure()
        if graph_type == "line":
            plt.plot(tik_values, gini_values, marker='o')
        else:  # 기본값은 막대 그래프
            plt.bar(tik_values, gini_values)
            
        plt.title(plt_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)  # 가독성을 위해 그리드 추가
        
        # 그래프를 타임스탬프 디렉토리에 저장
        plt.savefig(self.output_dir / 'result.png')

        # 저장된 파일 위치 출력
        output_msg = f"\nFiles saved in: {self.output_dir}\n"
        stream_content = {'name': 'stdout', 'text': output_msg}
        self.send_response(self.iopub_socket, 'stream', stream_content)

    def _handle_graph_creation(self, line):
        """
        그래프 생성 처리 함수
        """
        line = line.replace("is-a graph", "is-a prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type graph")
        self._execute_command(line)

    def _handle_agent_group_creation(self, line):
        """
        에이전트 그룹 생성 처리 함수
        """
        line = line.replace("agent-group", "prototype")
        self._execute_command(line, do_print=False)
        line = line.replace("is-a prototype", "type agent-group")
        self._execute_command(line)

    def _handle_get_variable(self, line):
        """
        변수 값을 가져오는 함수
        """
        match = re.search(r'\(get-variable (\w+) (\w+)\)', line)
        if match:
            class_name = match.group(1)
            variable_name = match.group(2)
        else:
            return

        if class_name in agent_list:
            temp = agent_list[class_name]
            
            if class_name not in var_return:
                var_return[class_name] = {}
            var_return[class_name][variable_name] = []

            for agent in temp:
                cmd = f"(get-list {agent} '{variable_name})"
                response = self._execute_command(cmd, do_print=False)
                var_return[class_name][variable_name].append(response.strip('"'))

            # 변수를 환경에 저장
            response = ' '.join(map(str, var_return[class_name][variable_name]))
            cmd = f"(set '{variable_name} '({response}))"
            self._execute_command(cmd)

    def _handle_save_data(self, line):
        """
        데이터 저장 처리 함수 (필요에 따라 구현)
        """
        pass  # 기능이 _handle_sim_start로 이동됨

    def _handle_message_box(self, line):
        """
        메시지 박스 처리 함수 (필요에 따라 구현)
        """
        pass  # 필요에 따라 구현

    def parse_input(input_string):
        """
        입력 문자열을 파싱하는 함수
        """
        inner_string = re.search(r'\(parse\s+\((.+)\)\)', input_string).group(1)
        pairs = re.findall(r'\((\w+)\s+(\w+)\)', inner_string)
        result_dict = {key: value for key, value in pairs}
        return result_dict

    def _handle_make_instance(self, line):
        """
        인스턴스 생성 처리 함수 (필요에 따라 구현)
        """
        pass  # 필요에 따라 구현

    def do_execute(self, code, silent, store_history=True,
                   user_expressions=None, allow_stdin=False):
        def execute_lines(line):
            temp.append(line)
            if "is-a agent)" in line:
                self._handle_agent_creation(line)
            elif "is-a grid)" in line:
                self._handle_grid_creation(line)
            elif "is-a map)" in line:
                self._handle_map_creation(line)
            elif line == "":
                pass
            elif "is-a rule)" in line:
                self._handle_rule_creation(line)
            elif "is-a AgentSimModel)" in line:
                self._handle_agent_sim_model_creation(line)
            elif "is-a region)" in line:
                self._handle_region_creation(line)
            elif "is-a graph)" in line:
                self._handle_graph_creation(line)
            elif "is-a agent-group)" in line:
                self._handle_agent_group_creation(line)
            elif "(gini function-call" in line:
                pass
            elif "(get-variable" in line:
                self._handle_get_variable(line)
            elif "(save-data" in line:
                self._handle_save_data(line)
            elif "(message-box" in line:
                self._handle_message_box(line)
            elif "(make-instance" in line:
                self._handle_make_instance(line)
            elif "(sim " in line:
                self._handle_sim_start(line)
            elif "(process_agent_removal" in line:
                pass  # 필요에 따라 구현
            elif line.startswith("(python '("):
                python_code = line[10:-2]

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                try:
                    exec(python_code)
                except Exception as e:
                    stream_content = {'name': 'stderr', 'text': str(e) + '\n'}
                    self.send_response(self.iopub_socket, 'stream', stream_content)
                
                sys.stdout = old_stdout
                
                output = mystdout.getvalue()
                
                stream_content = {'name': 'stdout', 'text': output}
                self.send_response(self.iopub_socket, 'stream', stream_content)

            elif "(parse " in line:
                result_dict = self._parse_input(line)
                response = str(result_dict)
                try:
                    with open('output.txt', 'r') as file:
                        output_contents = file.read()
                except FileNotFoundError:
                    output_contents = 'output.txt not found or unable to read.'
                stream_content = {'name': 'stdout', 'text': output_contents + '\n' + 'result: ' + response}
                self.send_response(self.iopub_socket, 'stream', stream_content)
            else:
                response = self._execute_command(line)
                self._write_to_log(response)

        if not silent:
            self._write_to_log(code)
            code_lines = code.split('\n')
            for line in code_lines:
                if "(load-kb " in line:
                    filename = line.strip()[10:-1]
                    self._write_to_log(filename)
                    try:
                        with open(filename, 'r') as file:
                            file_lines = file.readlines()
                        for fileline in file_lines:
                            if fileline.strip() == "":
                                continue
                            self._write_to_log(fileline)
                            execute_lines(fileline.strip())
                    except FileNotFoundError:
                        stream_content = {'name': 'stderr', 'text': f"File '{filename}' not found.\n"}
                        self.send_response(self.iopub_socket, 'stream', stream_content)
                        return {'status': 'error', 'execution_count': self.execution_count}
                else:
                    execute_lines(line.strip())

        return {'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }

if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=NeoConsoleKernel)
