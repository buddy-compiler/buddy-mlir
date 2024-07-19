lexer grammar FegenLexer;

fragment Schar: ~ ["\\\r\n];

fragment NONDIGIT: [a-zA-Z_];

fragment UPPERCASE: [A-Z];

fragment LOWERCASE: [a-z];

fragment ALLCASE: [a-zA-Z0-9_];

fragment NOZERODIGIT: [1-9];

fragment DIGIT: [0-9];

fragment SQuoteLiteral
    : '\'' (('\\' ([btnfr"'\\] | . |EOF))|( ~ ['\r\n\\]))* '\''
    ;

// literal

UnsignedInt: NOZERODIGIT DIGIT* | '0';

ScienceReal : (Plus | Minus)? UnsignedInt Dot UnsignedInt ( 'E' (Plus | Minus)? UnsignedInt )?;

ConstBoolean: 'true' | 'false';

// key words

FEGEN: 'fegen';

INPUTS: 'inputs';

RETURNS: 'returns';

ACTIONS: 'actions';

IR: 'ir';

OPERAND_VALUE: 'operandValue';

ATTRIBUTE_VALUE: 'attributeValue';

CPP_VALUE: 'cppValue';

OPERATION: 'operation';

FUNCTION: 'function';

TYPEDEF: 'typedef';

OPDEF: 'opdef';

ARGUMENTS: 'arguments';

RESULTS: 'results';

BODY: 'body';

EMPTY: 'null';

PARAMETERS: 'parameters';

ASSEMBLY_FORMAT: 'assemblyFormat';


// types
TYPE: 'Type';

TYPETEMPLATE: 'TypeTemplate';

BOOL: 'bool';

INT: 'int';

FLOAT: 'float';

DOUBLE: 'double';

// F64TENSOR: 'F64Tensor';

// F64VECTOR: 'F64Vector';

CHAR: 'char';

STRING: 'string';

LIST: 'list';

ANY: 'any';

OPTIONAL: 'optional';

INTEGER: 'Integer';

FLOATPOINT: 'FloatPoint';

TENSOR: 'Tensor';

VECTOR: 'Vector';

CPP: 'cpp';

OPERAND: 'operand';

ATTRIBUTE: 'attribute';

// stmt

IF: 'if';

ELSE: 'else';

FOR: 'for';

IN: 'in';

WHILE: 'while';

RETURN: 'return';

// identifiers

LexerRuleName: UPPERCASE (NONDIGIT | DIGIT)*;

ParserRuleName: LOWERCASE (NONDIGIT | DIGIT)*;

// literal

StringLiteral
    : SQuoteLiteral
    ;


// marks

AND: '&&';

Logic_OR: '||';

EQUAL: '==';

NOT_EQUAL: '!=';

Less: '<';

LessEqual: '<=';

Greater: '>';

GreaterEqual: '>=';

Comma: ',';

Semi: ';';

LeftParen: '(';

RightParen: ')';

LeftBracket: '[';

RightBracket: ']';

LeftBrace: '{';

RightBrace: '}';

Dot: '.';

Colon: ':';

OR: '|';

QuestionMark: '?';

Star: '*';

Div: '/';

Plus: '+';

Minus: '-';

Assign: '=';

Dollar: '$';

StarStar: '**';

MOD: '%';

Arror: '->';

Underline: '_';

Tilde: '~';

Exclamation: '!';

Range: '..';

BeginInclude: '@header' LeftBrace -> pushMode (TargetLanguageAction);

Whitespace: [ \t]+ -> skip;

Newline: ('\r' '\n'? | '\n') -> skip;

BlockComment: '/*' .*? '*/' -> skip;

LineComment: '//' ~ [\r\n]* -> skip;

mode TargetLanguageAction;

EndInclude: RightBrace -> popMode;

INCLUDE_CONTENT
    : .
    | '\n'
    | ' '
    ;

