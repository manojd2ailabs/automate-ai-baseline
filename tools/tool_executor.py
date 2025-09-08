


from typing import Optional, Dict, Any, List
import inspect

class ToolExecutor:
    """Simple tool executor with dynamic parameter handling"""
    
    def __init__(self, tools, shared_state):
        self.tools = {tool.name: tool for tool in tools}
        self.shared_state = shared_state
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with dynamic parameter resolution"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        tool = self.tools[tool_name]
        print(f"🛠️ Executing tool: {tool_name}")
        
        try:
            # Get tool parameters dynamically
            tool_input = self._resolve_tool_input(tool, kwargs)
            print(f"🔍 Tool input: {tool_input}")
            
            # Execute tool
            result = tool.invoke(tool_input)
            print(f"✅ Tool result: {str(result)[:100]}...")
            
            # Update shared state (preserve existing state)
            self._update_state(tool_name, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.shared_state[f"{tool_name}_error"] = error_msg
            raise e
    
    def _resolve_tool_input(self, tool, kwargs):
        """Dynamically resolve tool input from state and kwargs"""
        if not hasattr(tool, 'func'):
            value = next(iter(kwargs.values())) if kwargs else self.shared_state.get('file_path', '')
            return {'input': value} if isinstance(value, str) else value

        sig = inspect.signature(tool.func)
        params = list(sig.parameters.keys())
        
        if len(params) == 1:
            param_name = params[0]
            value = None

            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in self.shared_state:
                value = self.shared_state[param_name]
            elif 'input' in kwargs: # Fallback for 'input' from LLM
                value = kwargs['input']
            
            if value is not None:
                return {param_name: value}
            else:
                return kwargs # Let pydantic in the tool handle the missing parameter

        else:
            # Multi-parameter tool
            resolved_params = {}
            for param in params:
                if param in kwargs:
                    resolved_params[param] = kwargs[param]
                elif param in self.shared_state:
                    resolved_params[param] = self.shared_state[param]
            
            return resolved_params if resolved_params else kwargs
    
    def _update_state(self, tool_name: str, result):
        """Update shared state with tool result (preserve existing state)"""
        if isinstance(result, dict):
            # Only update new keys or overwrite existing ones, don't clear
            for key, value in result.items():
                self.shared_state[key] = value
            print(f"📝 Updated state with dict keys: {list(result.keys())}")
        else:
            self.shared_state[f"{tool_name}_result"] = result
            print(f"📝 Updated state: {tool_name}_result")