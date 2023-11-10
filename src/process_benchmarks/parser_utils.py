def parse_smtlib_to_simple_format(smtlib_input):
    lines = smtlib_input.strip().split('\n')
    variables = set()
    terminals = set()
    eq_list=[]
    print("Parse lines:")
    for line in lines:
        if line.startswith('(declare-fun') and line.endswith('() String)'):
            print(line)
            # Extracting variable
            var_name = line.split()[1].replace("()","")
            variables.add(var_name)
        elif line.startswith('(declare-const') and line.endswith('String)'):
            print(line)
            # Extracting variable
            var_name = line.split()[1]
            variables.add(var_name)
        elif line.startswith('(assert (='):
            print(line)
            # Extracting terminals
            terminals_list=parse_in_quotes(line)
            for t in terminals_list:
                for tt in t:
                    terminals.add(tt)
            # Extracting eq
            expression_elements=parse_in_parentheses(line)
            #print(expression_elements)
            lhs=expression_elements[0].replace("str.++ ","")
            rhs=expression_elements[1].replace("str.++ ","").replace("=","")
            lhs=lhs.split()
            rhs=rhs.split()
            print("lhs",lhs)
            print("rhs",rhs)
            eq_list.append([lhs,rhs])

    woorpje_variables,variable_mapping=replace_elements_with_alphabets(variables)
    print("----- parsed results -----")
    print("variable_mapping",variable_mapping)
    woorpje_equation_list=[]
    for eq in eq_list:
        print("---")
        print("lhs",eq[0])
        print("rhs", eq[1])
        lhs=eq[0]
        rhs=eq[1]
        replaced_lhs=[]
        for l in lhs:
            if l in variable_mapping.keys():
                replaced_lhs.append(variable_mapping[l])
            else:
                replaced_lhs.append(l)
        replaced_rhs = []
        for r in rhs:
            if r in variable_mapping.keys():
                replaced_rhs.append(variable_mapping[r])
            else:
                replaced_rhs.append(r)

        print("replaced_lhs",replaced_lhs)
        print("replaced_rhs",replaced_rhs)

        woorpje_equation="".join(replaced_lhs).replace("\"","") +"="+"".join(replaced_rhs).replace("\"","")
        print("woorpje_equation",woorpje_equation)
        woorpje_equation_list.append(woorpje_equation)
    woorpje_variables="".join(woorpje_variables)
    woorpje_terminals = "".join(terminals)
    return {
        'Variables': woorpje_variables,
        'Terminals': woorpje_terminals,
        'Equation': woorpje_equation_list
    }

def parse_in_quotes(text):
    # Splitting the string by double quotes
    parts = text.split('"')

    # Extracting the elements that were inside quotes
    quoted_parts = [parts[i] for i in range(1, len(parts), 2)]

    return quoted_parts


def parse_in_parentheses(text):
    stack = []
    result = []
    current = ""

    for char in text:
        if char == '(':
            # When an opening parenthesis is found, push to stack
            if current:
                stack.append(current)
                current = ""
            stack.append(char)
        elif char == ')':
            # When a closing parenthesis is found
            if current:
                stack.append(current)
                current = ""
            if stack:
                # Pop from stack and build the string inside this pair of parentheses
                content = ""
                while stack and stack[-1] != '(':
                    content = stack.pop() + content
                if stack:  # Pop the opening '('
                    stack.pop()
                result.append(content)
        else:
            current += char

    return result


def replace_elements_with_alphabets(original_set):
    # Capital alphabets
    alphabets = [chr(i) for i in range(65, 91)]  # ASCII values for A-Z

    # Ensure we have enough alphabets
    if len(original_set) > len(alphabets):
        raise ValueError("Not enough alphabets to replace all elements")

    # Create a mapping from original elements to alphabets
    mapping = dict(zip(original_set, alphabets))

    # Replace elements in the set
    replaced_set = {mapping[element] for element in original_set}

    return replaced_set, mapping
