
from Parser import Parser, EqParser, EqReader

def main():
    parser_type = EqParser
    file_path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/1.eq"


    reader=EqReader()
    content=reader.read(file_path)
    print(content)

    parser = Parser(parser_type)
    parsed_content=parser.parse(file_path)
    print(parsed_content)



if __name__ == '__main__':
    main()