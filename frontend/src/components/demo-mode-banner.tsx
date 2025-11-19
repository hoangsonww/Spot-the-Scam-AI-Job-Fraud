"use client";

import { useBackendStatus } from "@/lib/backend-status";
import { AlertCircle, RefreshCw, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function DemoModeBanner() {
  const { isConnected, isChecking, error, checkConnection } = useBackendStatus();

  if (isConnected) {
    return null;
  }

  return (
    <Alert className="mb-6 border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-950">
      <Info className="h-4 w-4 text-blue-600 dark:text-blue-400" />
      <AlertTitle className="text-blue-900 dark:text-blue-100">Demo Mode Active</AlertTitle>
      <AlertDescription className="text-blue-800 dark:text-blue-200">
        <div className="space-y-2">
          <p>
            The backend server is not connected. This application is running in{" "}
            <strong>Demo Mode</strong> with realistic mock data to showcase its capabilities.
          </p>
          <p className="text-sm">
            All features are functional and use simulated responses that mirror the actual system's
            behavior. To enable full functionality with the live ML model, start the backend server
            and refresh this page.
          </p>
          <div className="mt-3 flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={checkConnection}
              disabled={isChecking}
              className="border-blue-600 text-blue-600 hover:bg-blue-100 dark:border-blue-400 dark:text-blue-400 dark:hover:bg-blue-900"
            >
              <RefreshCw className={`mr-2 h-3 w-3 ${isChecking ? "animate-spin" : ""}`} />
              {isChecking ? "Checking..." : "Check Connection"}
            </Button>
            {error && (
              <span className="text-xs text-blue-700 dark:text-blue-300">
                <AlertCircle className="mr-1 inline h-3 w-3" />
                {error}
              </span>
            )}
          </div>
        </div>
      </AlertDescription>
    </Alert>
  );
}
