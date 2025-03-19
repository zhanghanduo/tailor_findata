import inspect
import sys
import traceback
from rag_system.models.hybrid_retriever import HybridFinancialRetriever

class DebugRetriever(HybridFinancialRetriever):
    # Override the method to add debugging
    def _retrieve_from_all_indices(self, *args, **kwargs):
        print("Calling _retrieve_from_all_indices with:")
        print("Args:", args)
        print("Kwargs:", kwargs)
        return super()._retrieve_from_all_indices(*args, **kwargs)
    
    def _retrieve_from_index_hybrid(self, *args, **kwargs):
        print("Calling _retrieve_from_index_hybrid with:")
        print("Args:", args)
        print("Kwargs:", kwargs)
        try:
            return super()._retrieve_from_index_hybrid(*args, **kwargs)
        except Exception as e:
            print("Error in _retrieve_from_index_hybrid:", str(e))
            traceback.print_exc()
            raise

# Let's look at the source code of the retrieve method
print("Source code of retrieve method:")
print(inspect.getsource(HybridFinancialRetriever.retrieve))

print("\nSource code of _retrieve_from_all_indices method:")
print(inspect.getsource(HybridFinancialRetriever._retrieve_from_all_indices))

print("\nSource code of _retrieve_from_index_hybrid method:")
print(inspect.getsource(HybridFinancialRetriever._retrieve_from_index_hybrid))

# Now, let's print the line numbers of these functions to see where they are defined
print("\nLine numbers:")
print("HybridFinancialRetriever.retrieve:", inspect.getsourcelines(HybridFinancialRetriever.retrieve)[1])
print("HybridFinancialRetriever._retrieve_from_all_indices:", inspect.getsourcelines(HybridFinancialRetriever._retrieve_from_all_indices)[1])
print("HybridFinancialRetriever._retrieve_from_index_hybrid:", inspect.getsourcelines(HybridFinancialRetriever._retrieve_from_index_hybrid)[1]) 