
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def get(**kwargs):
        import streamlit.report_thread as ReportThread
        from streamlit.server.server import Server
        
        ctx = ReportThread.get_report_ctx().session_id
        this_session = None

        current_server = Server.get_current()
        if hasattr(current_server, "_session_info_by_id"):
            session_infos = current_server._session_info_by_id.values()
        else:
            session_infos = current_server._session_info.values()

        for session_info in session_infos:
            if session_info.session.session_id == ctx:
                this_session = session_info.session

        if this_session is None:
            raise RuntimeError("Couldn't get your Streamlit Session object.")

        if not hasattr(this_session, "_custom_session_state"):
            this_session._custom_session_state = SessionState(**kwargs)

        return this_session._custom_session_state
