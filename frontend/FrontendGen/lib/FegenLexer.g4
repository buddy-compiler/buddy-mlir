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

OPERAND_VALUE: 'operandValue';

ATTRIBUTE_VALUE: 'attributeValue';

CPP_VALUE: 'cppValue';

LIST: 'list';

OPERATION: 'operation';

ATTRIBUTE: 'attribute';

FUNCTION: 'function';

TYPE_IR: 'typeDef';

INT: 'int';

FLOAT: 'float';

TENSOR: 'tensor';

STRING: 'string';

NULL: 'null';




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

Whitespace: [ \t]+ -> skip;

Newline: ('\r' '\n'? | '\n') -> skip;

BlockComment: '/*' .*? '*/' -> skip;

LineComment: '//' ~ [\r\n]* -> skip;