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

// key words

FEGEN: 'fegen';

INPUTS: 'inputs';

RETURNS: 'returns';

ACTIONS: 'actions';

IR: 'ir';

PARAMETERS: 'parameters';

ASSEMBLYFORMAT: 'assemblyFormat';

OPERAND_VALUE: 'operandValue';

ATTRIBUTE_VALUE: 'attributeValue';

CPP_VALUE: 'cppValue';

LIST: 'list';

OPERATION: 'operation';

ATTRIBUTE: 'Attribute';

TYPE: 'Type';

FUNCTION: 'function';

TYPEDEF: 'typedef';

ATTRIBUTEDEF: 'attributedef';

INT: 'int';

FLOAT: 'float';

DOUBLE: 'double';

TENSOR: 'tensor';

STRING: 'string';

INTATTR: 'intAttr';

FLOATATTR: 'floatAttr';

DOUBLEATTR: 'doubleAttr';

TENSORATTR: 'tensorAttr';

STRINGATTR: 'stringAttr';

NULL: 'null';

// AnyTypeOf method
ANYTYPEOF: 'AnyTypeOf';

//assmblyFormat identifier

AssmblyIden: BackQuote Less BackQuote (NONDIGIT | Less | Greater | LeftParen | RightParen | Dollar)* BackQuote Less BackQuote;

// identifiers

LexerRuleName: UPPERCASE (NONDIGIT | DIGIT)*;

ParserRuleName: LOWERCASE (NONDIGIT | DIGIT)*;

// literal

StringLiteral
    : SQuoteLiteral
    ;

IntLiteral
    : '0' | NOZERODIGIT DIGIT*
    ;


// marks

Less: '<';

Greater: '>';

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

Range: '..';

BackQuote: '`';



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

