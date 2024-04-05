from langchain.schema import(
    HumanMessage, 
    SystemMessage
)

class Messages:
    @staticmethod
    def get_dfs_employee_messages(question, sql_query_result):
        messages = [
            SystemMessage(content='You are an expert on providing Dell Financial System information. Give a precise and accurate answer to the question below. Always provide the in the most polite way possible.Do not provide any additional information.'),
            HumanMessage(content=f'Provide the answer to the question {question} based on the following value:\n TEXT: {sql_query_result}')
        ]
        return messages

    @staticmethod
    def get_operation_type_messages(operationTypesQueryResult, query):
        messages = [
            SystemMessage(content='You are an expert in getting the correct operation type based on the user question. Only provide the operation type that is most similar to the one in the question itself. Do not provide any additional information.'),
            HumanMessage(content=f'Provide the most similar type name among all the following types: {operationTypesQueryResult} ,for the provided question {query}')
        ]
        return messages
    
    @staticmethod
    def get_count_operation_messages(operation_type, uncheckedSqlQueryChain):
        messages = [
            SystemMessage(content='You are an expert on correcting the microsoft sql server query. Only provide the query itself. Do not provide any additional information.'),
            HumanMessage(content=f'Provide the correct query. OPERTN_NM should be equal to {operation_type} in the where clause on the following query: {uncheckedSqlQueryChain}')
        ]
        return messages